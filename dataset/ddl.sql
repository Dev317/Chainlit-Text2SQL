CREATE TABLE `bigquery-public-data.ml_datasets.census_adult_income`
(
  age INT64 OPTIONS(description="Age"),
  workclass STRING OPTIONS(description="Nature of employment"),
  functional_weight INT64 OPTIONS(description="Sample weight of the individual from the original Census data. How likely they were to be included in this dataset, based on their demographic characteristics vs. whole-population estimates."),
  education STRING OPTIONS(description="Level of education completed"),
  education_num INT64 OPTIONS(description="Estimated years of education completed based on the value of the education field."),
  marital_status STRING OPTIONS(description="Marital status"),
  occupation STRING OPTIONS(description="Occupation category"),
  relationship STRING OPTIONS(description="Relationship to the household"),
  race STRING OPTIONS(description="Race"),
  sex STRING OPTIONS(description="Gender"),
  capital_gain INT64 OPTIONS(description="Amount of capital gains"),
  capital_loss INT64 OPTIONS(description="Amount of capital loss"),
  hours_per_week INT64 OPTIONS(description="Hours worked per week"),
  native_country STRING OPTIONS(description="Country of birth"),
  income_bracket STRING OPTIONS(description="Either \">50K\" or \"<=50K\" based on income.")
);