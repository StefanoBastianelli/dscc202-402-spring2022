# Databricks notebook source
# MAGIC %md
# MAGIC # Putting it all together: Managing the Machine Learning Lifecycle
# MAGIC 
# MAGIC Create a workflow that includes pre-processing logic, the optimal ML algorithm and hyperparameters, and post-processing logic.
# MAGIC 
# MAGIC ## Instructions
# MAGIC 
# MAGIC In this course, we've primarily used Random Forest in `sklearn` to model the Airbnb dataset.  In this exercise, perform the following tasks:
# MAGIC <br><br>
# MAGIC 0. Create custom pre-processing logic to featurize the data
# MAGIC 0. Try a number of different algorithms and hyperparameters.  Choose the most performant solution
# MAGIC 0. Create related post-processing logic
# MAGIC 0. Package the results and execute it as its own run
# MAGIC 
# MAGIC ## Prerequisites
# MAGIC - Web browser: Chrome
# MAGIC - A cluster configured with **8 cores** and **DBR 7.0 ML**

# COMMAND ----------

# MAGIC %md
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Classroom-Setup
# MAGIC 
# MAGIC For each lesson to execute correctly, please make sure to run the **`Classroom-Setup`** cell at the<br/>
# MAGIC start of each lesson (see the next cell) and the **`Classroom-Cleanup`** cell at the end of each lesson.

# COMMAND ----------

# MAGIC %run "./Includes/Classroom-Setup"

# COMMAND ----------

# Adust our working directory from what DBFS sees to what python actually sees
working_path = workingDir.replace("dbfs:", "/dbfs")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pre-processing
# MAGIC 
# MAGIC Take a look at the dataset and notice that there are plenty of strings and `NaN` values present. Our end goal is to train a sklearn regression model to predict the price of an airbnb listing.
# MAGIC 
# MAGIC 
# MAGIC Before we can start training, we need to pre-process our data to be compatible with sklearn models by making all features purely numerical. 

# COMMAND ----------

import pandas as pd

airbnbDF = spark.read.parquet("/mnt/training/airbnb/sf-listings/sf-listings-correct-types.parquet").toPandas()

display(airbnbDF)

# COMMAND ----------

# MAGIC %md
# MAGIC In the following cells we will walk you through the most basic pre-processing step necessary. Feel free to add additional steps afterwards to improve your model performance.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC First, convert the `price` from a string to a float since the regression model will be predicting numerical values.

# COMMAND ----------

# TODO
                  
airbnbDF["price"] = airbnbDF["price"].str.replace("$","")
airbnbDF["price"] = airbnbDF["price"].str.replace(",","")
airbnbDF['price'] = airbnbDF['price'].astype(float)
            

# COMMAND ----------

# MAGIC %md
# MAGIC Take a look at our remaining columns with strings (or numbers) and decide if you would like to keep them as features or not.
# MAGIC 
# MAGIC Remove the features you decide not to keep.

# COMMAND ----------

# TODO
airbnbDF = airbnbDF.drop(['host_is_superhost', 'cancellation_policy','instant_bookable'], axis=1)

# COMMAND ----------

airbnbDF = airbnbDF.round({"latitude":2, "longitude":2})



airbnbDF["review_scores_sum"] = airbnbDF['review_scores_accuracy'] + airbnbDF['review_scores_cleanliness'] + airbnbDF['review_scores_checkin'] + airbnbDF['review_scores_communication'] + airbnbDF['review_scores_location'] +airbnbDF['review_scores_value']

airbnbDF = airbnbDF.drop(['review_scores_accuracy', 'review_scores_cleanliness','review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value'], axis=1)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC For the string columns that you've decided to keep, pick a numerical encoding for the string columns. Don't forget to deal with the `NaN` entries in those columns first.

# COMMAND ----------

from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
encoder_df = pd.DataFrame(encoder.fit_transform(airbnbDF[['property_type','room_type', 'bed_type']]).toarray())
airbnbDF = airbnbDF.join(encoder_df)
airbnbDF = airbnbDF.dropna()

airbnbDF = airbnbDF.drop(['neighbourhood_cleansed', 'property_type','room_type', 'bed_type'], axis=1)

# COMMAND ----------

airbnbDF

# COMMAND ----------

# MAGIC %md
# MAGIC Before we create a train test split, check that all your columns are numerical. Remember to drop the original string columns after creating numerical representations of them.
# MAGIC 
# MAGIC Make sure to drop the price column from the training data when doing the train test split.

# COMMAND ----------

# TODO
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(airbnbDF.drop(["price"], axis=1), airbnbDF[["price"]].values.ravel(), random_state=42)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model
# MAGIC 
# MAGIC After cleaning our data, we can start creating our model!

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Firstly, if there are still `NaN`'s in your data, you may want to impute these values instead of dropping those entries entirely. Make sure that any further processing/imputing steps after the train test split is part of a model/pipeline that can be saved.
# MAGIC 
# MAGIC In the following cell, create and fit a single sklearn model.

# COMMAND ----------

# TODO
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

rf = RandomForestRegressor(n_estimators=120, max_depth=50)
rf.fit(X_train, y_train)
rf_mse = mean_squared_error(y_test, rf.predict(X_test))

r2_score(y_test, rf.predict(X_test))

# COMMAND ----------

# MAGIC %md
# MAGIC Pick and calculate a regression metric for evaluating your model.

# COMMAND ----------

# TODO

rf_mse = mean_squared_error(y_test, rf.predict(X_test))
r2 = r2_score(y_test, rf.predict(X_test))
rf_mae = mean_absolute_error(y_test, rf.predict(X_test))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Log your model on MLflow with the same metric you calculated above so we can compare all the different models you have tried! Make sure to also log any hyperparameters that you plan on tuning!

# COMMAND ----------

# TODO

import mlflow.sklearn
with mlflow.start_run(run_name="RF_final_model") as run: 
  mlflow.sklearn.log_model(rf, "random-forest-model")
  mlflow.log_metric("mse", rf_mse)
  mlflow.log_metric("r2", r2)
  mlflow.log_metric("mae", rf_mae)
  mlflow.log_param("n_estimators", 120)
  mlflow.log_param("max_depth", 50)
    

  
  
  experimentID = run.info.experiment_id
  artifactURI = mlflow.get_artifact_uri()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Change and re-run the above 3 code cells to log different models and/or models with different hyperparameters until you are satisfied with the performance of at least 1 of them.

# COMMAND ----------

from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X_train, y_train)
reg_mse = mean_squared_error(y_test, reg.predict(X_test))
reg_r2 = r2_score(y_test, reg.predict(X_test))
reg_mae = mean_absolute_error(y_test, reg.predict(X_test))

from sklearn.linear_model import TweedieRegressor
reg_t = TweedieRegressor(max_iter = 1000, alpha = 0.5)
reg_t.fit(X_train, y_train)
reg_t_mse = mean_squared_error(y_test, reg_t.predict(X_test))
reg_tr2 = r2_score(y_test, reg_t.predict(X_test))
reg_tmae = mean_absolute_error(y_test, reg_t.predict(X_test))



from sklearn import linear_model
bay = linear_model.BayesianRidge(n_iter=1000, tol=0.001, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06)
bay.fit(X_train, y_train)
bay_mse = mean_squared_error(y_test, bay.predict(X_test))
bay_r2 = r2_score(y_test, bay.predict(X_test))
bay_mae = mean_absolute_error(y_test, bay.predict(X_test))

with mlflow.start_run(run_name="linear_reg") as run: 
  mlflow.sklearn.log_model(reg, "linear_reg-model")
  mlflow.log_metric("mse", reg_mse)
  mlflow.log_metric("r2", reg_r2)
  mlflow.log_metric("reg_mae", rf_mae)
  
  experimentID = run.info.experiment_id
  artifactURI1 = mlflow.get_artifact_uri()


with mlflow.start_run(run_name="reg_t") as run: 
  mlflow.sklearn.log_model(reg_t, "reg_t-model")
  mlflow.log_metric("mse", reg_t_mse)
  mlflow.log_metric("r2", reg_tr2)
  mlflow.log_metric("reg_mae", reg_tmae)
  mlflow.log_param("max_iter", 300)
  mlflow.log_param("alpha", 0.5)
  
  experimentID = run.info.experiment_id
  artifactURI2 = mlflow.get_artifact_uri()

with mlflow.start_run(run_name="BayesianRidge") as run: 
  mlflow.sklearn.log_model(bay, "BayesianRidge-model")
  mlflow.log_metric("mse", bay_mse)
  mlflow.log_metric("r2", bay_r2)
  mlflow.log_metric("reg_mae", bay_mae)
  mlflow.log_param("max_iter", 300)
  mlflow.log_param("tol", 0.001)
  mlflow.log_param("alpha_1", 1e-06)
  mlflow.log_param("alpha_2", 1e-06)
  mlflow.log_param("lambda_1", 1e-06)
  mlflow.log_param("lambda_2", 1e-06)
    
  experimentID = run.info.experiment_id
  artifactURI3 = mlflow.get_artifact_uri()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Look through the MLflow UI for the best model. Copy its `URI` so you can load it as a `pyfunc` model.

# COMMAND ----------

# TODO
import mlflow.pyfunc
from  mlflow.tracking import MlflowClient

client = MlflowClient()
#reg_run = sorted(client.list_run_infos(experimentID), key=lambda r: r.start_time, reverse=True)[0]
#reg_path = rf2_run.artifact_uri+"/random-forest-model-preprocess/"
rf2_pyfunc_model = mlflow.pyfunc.load_model('dbfs:/databricks/mlflow-tracking/1616927778606363/c72f128cd03f4ab48f974b9e74af0ee0/artifacts/linear_reg-model')
#rf2_pyfunc_model = mlflow.pyfunc.load_pyfunc(rf2_path.replace("dbfs:", "/dbfs"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Post-processing
# MAGIC 
# MAGIC Our model currently gives us the predicted price per night for each Airbnb listing. Now we would like our model to tell us what the price per person would be for each listing, assuming the number of renters is equal to the `accommodates` value. 

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC Fill in the following model class to add in a post-processing step which will get us from total price per night to **price per person per night**.
# MAGIC 
# MAGIC <img alt="Side Note" title="Side Note" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.05em; transform:rotate(15deg)" src="https://files.training.databricks.com/static/images/icon-note.webp"/> Check out <a href="https://www.mlflow.org/docs/latest/models.html#id13" target="_blank">the MLFlow docs for help.</a>

# COMMAND ----------

# TODO

class Airbnb_Model(mlflow.pyfunc.PythonModel):

    def __init__(self, model):
        self.reg = model
    
    def predict(self, model_input):
        
        results = self.reg.predict(model_input)
        return results



# COMMAND ----------

# MAGIC %md
# MAGIC Construct and save the model to the given `final_model_path`.

# COMMAND ----------

# TODO
final_model_path =  f"{working_path}/final-model"

dbutils.fs.rm(final_model_path, True)

reg_postprocess_model = Airbnb_Model(model = reg)
mlflow.pyfunc.save_model(path=final_model_path.replace("dbfs:", "/dbfs"), python_model=reg_postprocess_model)





# COMMAND ----------

# MAGIC %md
# MAGIC Load the model in `python_function` format and apply it to our test data `X_test` to check that we are getting price per person predictions now.

# COMMAND ----------

# TODO
loaded_preprocess_model = mlflow.pyfunc.load_pyfunc('dbfs:/databricks/mlflow-tracking/1616927778606363/c72f128cd03f4ab48f974b9e74af0ee0/artifacts/linear_reg-model')

loaded_preprocess_model.predict(X_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Packaging your Model
# MAGIC 
# MAGIC Now we would like to package our completed model! 

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC First save your testing data at `test_data_path` so we can test the packaged model.
# MAGIC 
# MAGIC <img alt="Hint" title="Hint" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.3em" src="https://files.training.databricks.com/static/images/icon-light-bulb.svg"/>&nbsp;**Hint:** When using `.to_csv` make sure to set `index=False` so you don't end up with an extra index column in your saved dataframe.

# COMMAND ----------

# TODO
save the testing data 
test_data_path = f"{working_path}/test_data.csv"
# FILL_IN

prediction_path = f"{working_path}/predictions.csv"

# COMMAND ----------

# MAGIC %md
# MAGIC First we will determine what the project script should do. Fill out the `model_predict` function to load out the trained model you just saved (at `final_model_path`) and make price per person predictions on the data at `test_data_path`. Then those predictions should be saved under `prediction_path` for the user to access later.
# MAGIC 
# MAGIC Run the cell to check that your function is behaving correctly and that you have predictions saved at `demo_prediction_path`.

# COMMAND ----------

# TODO
import click
import mlflow.pyfunc
import pandas as pd

@click.command()
@click.option("--final_model_path", default="", type=str)
@click.option("--test_data_path", default="", type=str)
@click.option("--prediction_path", default="", type=str)
def model_predict(final_model_path, test_data_path, prediction_path):
    # FILL_IN


# test model_predict function    
demo_prediction_path = f"{working_path}/predictions.csv"

from click.testing import CliRunner
runner = CliRunner()
result = runner.invoke(model_predict, ['--final_model_path', final_model_path, 
                                       '--test_data_path', test_data_path,
                                       '--prediction_path', demo_prediction_path], catch_exceptions=True)

assert result.exit_code == 0, "Code failed" # Check to see that it worked
print("Price per person predictions: ")
print(pd.read_csv(demo_prediction_path))

# COMMAND ----------

# MAGIC %md
# MAGIC Next, we will create a MLproject file and put it under our `workingDir`. Complete the parameters and command of the file.

# COMMAND ----------

# TODO
dbutils.fs.put(f"{workingDir}/MLproject", 
'''
name: Capstone-Project

conda_env: conda.yaml

entry_points:
  main:
    parameters:
      #FILL_IN
    command:  "python predict.py #FILL_IN"
'''.strip(), overwrite=True)

# COMMAND ----------

print(prediction_path)

# COMMAND ----------

# MAGIC %md
# MAGIC We then create a `conda.yaml` file to list the dependencies needed to run our script.
# MAGIC 
# MAGIC For simplicity, we will ensure we use the same version as we are running in this notebook.

# COMMAND ----------

import cloudpickle, numpy, pandas, sklearn, sys

version = sys.version_info # Handles possibly conflicting Python versions

file_contents = f"""
name: Capstone
channels:
  - defaults
dependencies:
  - python={version.major}.{version.minor}.{version.micro}
  - cloudpickle={cloudpickle.__version__}
  - numpy={numpy.__version__}
  - pandas={pandas.__version__}
  - scikit-learn={sklearn.__version__}
  - pip:
    - mlflow=={mlflow.__version__}
""".strip()

dbutils.fs.put(f"{workingDir}/conda.yaml", file_contents, overwrite=True)

print(file_contents)

# COMMAND ----------

# MAGIC %md
# MAGIC Now we will put the **`predict.py`** script into our project package.
# MAGIC 
# MAGIC Complete the **`.py`** file by copying and placing the **`model_predict`** function you defined above.

# COMMAND ----------

# TODO
dbutils.fs.put(f"{workingDir}/predict.py", 
'''
import click
import mlflow.pyfunc
import pandas as pd

# put model_predict function with decorators here
    
if __name__ == "__main__":
  model_predict()

'''.strip(), overwrite=True)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's double check all the files we've created are in the `workingDir` folder. You should have at least the following 3 files:
# MAGIC * `MLproject`
# MAGIC * `conda.yaml`
# MAGIC * `predict.py`

# COMMAND ----------

display( dbutils.fs.ls(workingDir) )

# COMMAND ----------

# MAGIC %md
# MAGIC Under **`workingDir`** is your completely packaged project.
# MAGIC 
# MAGIC Run the project to use the model saved at **`final_model_path`** to predict the price per person of each Airbnb listing in **`test_data_path`** and save those predictions under **`second_prediction_path`** (defined below).

# COMMAND ----------

# TODO
second_prediction_path = f"{working_path}/predictions-2.csv"
mlflow.projects.run(working_path,
   # FILL_IN
)

# COMMAND ----------

# MAGIC %md
# MAGIC Run the following cell to check that your model's predictions are there!

# COMMAND ----------

print("Price per person predictions: ")
print(pd.read_csv(second_prediction_path))

# COMMAND ----------

# MAGIC %md
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Classroom-Cleanup<br>
# MAGIC 
# MAGIC Run the **`Classroom-Cleanup`** cell below to remove any artifacts created by this lesson.

# COMMAND ----------

# MAGIC %run "./Includes/Classroom-Cleanup"

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC <h2><img src="https://files.training.databricks.com/images/105/logo_spark_tiny.png"> All done!</h2>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2020 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
