import os
import chainlit as cl

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI, ChatVertexAI
import pandas as pd
import io
from langchain.callbacks.base import AsyncCallbackHandler
from langchain.document_loaders.csv_loader import CSVLoader
import shutil
import logging
from chainlit.input_widget import TextInput
import traceback
from langchain.llms.vertexai import VertexAI
from langchain.embeddings.vertexai import VertexAIEmbeddings
from chainlit.input_widget import Slider

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from langfuse.callback import CallbackHandler
from langfuse import Langfuse
from langfuse.model import CreateScore
from langfuse.model import CreateScoreRequest

ENV_HOST = os.getenv("LANGFUSE_HOST")
ENV_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
ENV_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")

handler = CallbackHandler(ENV_PUBLIC_KEY, ENV_SECRET_KEY, ENV_HOST)

answer_prefix_tokens=["FINAL", "ANSWER"]
if os.path.exists('./chroma'):
    shutil.rmtree("./chroma")

class CustomStuffDocumentsChainHandler(AsyncCallbackHandler):
    async def on_retriever_end(
        self,
        documents,
        *,
        run_id,
        parent_run_id,
        tags,
        **kwargs,
    ) -> None:
        """Run on retriever end."""
        elements = []
        for doc in documents:
            content = f"Content: {doc.page_content}"
            elements.append(cl.Text(content=content, display="inline", name=f"{doc.metadata['source']}"))
        await cl.Message(content="Sources:", author="StuffDocumentsChain", elements=elements, indent=2).send()

class CustomLLMChainHandler(AsyncCallbackHandler):
    async def on_llm_start(
        self,
        serialized,
        prompts,
        *,
        run_id,
        parent_run_id = None,
        tags = None,
        metadata = None,
        **kwargs,
    ) -> None:
        """Run when LLM starts running."""
        await cl.Message(content="Prompt:\n" + " ".join(prompts), author="LLMChain", indent=3).send()

class CustomConversationalRetrievalQAHandler(AsyncCallbackHandler):
    async def on_llm_start(
        self,
        serialized,
        prompts,
        *,
        run_id,
        parent_run_id = None,
        tags = None,
        metadata =None,
        **kwargs,
    ) -> None:
        """Run when LLM starts running."""
        await cl.Message(content="Prompt:\n" + " ".join(prompts), author="ConversationalRetrievalChain", indent=1).send()

async def process_file(file):
    import tempfile

    # Decode the file
    text = file.content.decode("utf-8")

    df = pd.read_csv(io.StringIO(text))
    data_message = cl.Message(content="Displaying first 5 rows:\n" + df.head(5).to_markdown())
    await data_message.send()

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(file.content)
        tmp_file_path = tmp_file.name

    loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={'delimiter': ','})
    data = loader.load()

    # Create a metadata for each chunk
    metadatas = [{"source": f"Row-{i}"} for i in range(len(data))]
    data_str = [doc.page_content for doc in data]

    # Create a Chroma vector store
    # embeddings = VertexAIEmbeddings()
    embeddings = OpenAIEmbeddings()
    return await cl.make_async(Chroma.from_texts)(
        data_str, embeddings, metadatas=metadatas
    )

async def get_chain(docsearch):
    condense_question_template = """Considering the provided chat history and a subsequent question, rewrite the follow-up question to be an independent query. Alternatively, conclude the conversation if it appears to be complete.
Chat History:\"""
{chat_history}
\"""
Follow Up Input: \"""
{question}
\"""
Standalone question:"""

    qa_template = """
You're an AI assistant specializing in data analysis with Google SQL.
Based on the question provided, if it pertains to data analysis or Google SQL tasks, generate SQL code that is compatible with the Google SQL environment.
Additionally, offer a brief explanation about how you arrived at the Google SQL code.
Do not create undefined columns. You are only allowed to query the database.
Do use the data below that is provided to generate accurate Google SQL. These rows are meant as a way of telling you about the column data type and schema of the table.
{context}

**You are only required to write one SQL query per question.**
If the question or context does not clearly involve Google SQL or data analysis tasks, respond appropriately without generating Google SQL queries.
When the user expresses gratitude or says "Thanks", interpret it as a signal to conclude the conversation. Respond with an appropriate closing statement without generating further SQL queries.
If you don't know the answer, simply state, "I'm sorry, I don't know the answer to your question."
Write your response in markdown format and code in ```sql```.
Do use the DDL schema from the question to generate accurate Google SQL.
Question: ```{question}```

Answer:
"""
    # LLM = VertexAI(model='codechat-bison',
    #                max_output_tokens=2048)
    LLM = ChatOpenAI(model='gpt-3.5-turbo-16k', temperature=0.5)
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_question_template)
    QA_PROMPT = PromptTemplate(template=qa_template, input_variables=["question", "context"])
    question_generator = LLMChain(llm=LLM,
                                  prompt=CONDENSE_QUESTION_PROMPT,
                                )

    doc_chain = load_qa_chain(llm=LLM,
                              chain_type="stuff",
                              prompt=QA_PROMPT,
                            )

    conv_chain = ConversationalRetrievalChain(
        retriever=docsearch.as_retriever(),
        combine_docs_chain=doc_chain,
        question_generator=question_generator,
        return_source_documents=True,
        return_generated_question=True,
        rephrase_question=False,
    )
    return conv_chain



@cl.on_chat_start
async def init():
    await cl.ChatSettings(
        [
            TextInput(id="feedback_input", label="Feedback Input", tooltip="Provide feedback on current LLM response"),
            Slider(
                id="score",
                label="LLM Response Score",
                initial=0,
                min=-1,
                max=1,
                step=0.1,
                description="Score for an LLM response",
                tooltip="-1 is Bad, 1 is Good"
            ),
        ]
    ).send()

    files = None

    # Wait for the user to upload a file
    while files == None:
        files = await cl.AskFileMessage(
            content="Please upload a data sample csv file to begin!",
            accept=["text/csv"],
            max_files = 1
        ).send()

    csv_file = files[0]

    processing_msg = cl.Message(content=f"Processing csv file")
    await processing_msg.send()

    docsearch = await process_file(csv_file)

    # Create a chain that uses the Chroma vector store
    chain = await get_chain(docsearch)

    # Let the user know that the system is ready
    processing_complete_msg = cl.Message(f"Processing `{csv_file.name}` done!")
    await processing_complete_msg.send()

    files = None
    while files == None:
        files = await cl.AskFileMessage(
            content="Please upload a DDL schema file to begin!",
            accept={"text/plain": [".sql"]}
        ).send()
    sql_file = files[0]
    sql = sql_file.content.decode("utf-8")
    cl.user_session.set("sql", sql)

    processing_sql_message = cl.Message(content=f"Processing sql file")
    await processing_sql_message.send()

    sql_msg = cl.Message(content=sql, language='sql')
    await sql_msg.send()

    processing_complete_msg.content = cl.Message(content=f"Processing `{sql_file.name}` done!\n Now you can start asking question!")
    await processing_complete_msg.send()

    cl.user_session.set("chain", chain)
    cl.user_session.set("history", [])


@cl.on_message
async def main(message: str):
    chain = cl.user_session.get("chain")
    history = cl.user_session.get("history")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True,
        answer_prefix_tokens=answer_prefix_tokens,
    )

    qa_prompt = message + f'\nThe DDL schema is: \n {cl.user_session.get("sql")}'

    try:
        result = await chain.acall({"question": qa_prompt,
                        "chat_history": history},
                        callbacks=[cb, handler])

        answer = result["answer"]
        source_documents = result['source_documents']
        content = [source_documents[i].page_content for i in range(len(source_documents))]
        name = [source_documents[i].metadata['source'] for i in range(len(source_documents))]
        source_elements = [
            cl.Text(content=content[i], name=name[i]) for i in range(len(source_documents))
        ]

        if source_documents:
            answer += f"\n\nSources: {', '.join([source_documents[i].metadata['source'] for i in range(len(source_documents))])}"
        else:
            answer += "\n\nNo sources found"

        if cb.has_streamed_final_answer:
            cb.final_stream.elements = source_elements
            await cb.final_stream.update()
        else:
            history.append((message, result["answer"]))
            cl.user_session.set("history", history)
            answer_msg = cl.Message(content="", elements=source_elements)

            for i in answer:
                await answer_msg.stream_token(i)

            await answer_msg.send()

        cl.user_session.set('user_question', message)
        cl.user_session.set('llm_response', result["answer"])
        handler.langfuse.flush()

    except Exception as ex:
        logging.error(f"Error: {str(ex)}")
        logging.debug(f"{traceback.format_exc()}")


@cl.on_settings_update
def log_feedback(settings):
    feedback = ""
    score = 0
    if "feedback_input" in settings:
        feedback = settings["feedback_input"]
        logging.info(f"Feedback:\n\nUser Question:{cl.user_session.get('user_question', '')}\n\nLLM Response:{cl.user_session.get('llm_response', '')}\n\nUser Feedback:{feedback}")

    if "score" in settings:
        traceId = handler.get_trace_id()

        # Add score, e.g. via the Python SDK
        score = settings["score"]
        logging.info(f"Response score: {score}")

    langfuse = Langfuse(ENV_PUBLIC_KEY, ENV_SECRET_KEY, ENV_HOST)
    trace = langfuse.score(
        CreateScoreRequest(
            traceId=traceId,
            name="user-feedback",
            value=score,
            comment=feedback
        )
    )

    logging.info(f"Send score: {trace}")

