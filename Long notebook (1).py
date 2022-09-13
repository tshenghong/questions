# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Single Node Data Science on Databricks test1234
# MAGIC 
# MAGIC <img src="https://hackr.io/blog/top-data-science-python-libraries/thumbnail/large", width =400>
# MAGIC <img src="https://redislabs.com/wp-content/uploads/2016/12/lgo-partners-databricks.png",  width =240>
# MAGIC 
# MAGIC 
# MAGIC **Contents**
# MAGIC 
# MAGIC * Databricks ML Runtime
# MAGIC * Centralized data management
# MAGIC * Visualization
# MAGIC * Native ML Tools in Databricks
# MAGIC   * SQL
# MAGIC   * MLflow
# MAGIC   * Scaled Hyperparameter Tuning
# MAGIC   * Koalas
# MAGIC * Package Management
# MAGIC   * Notebook scoped, cluster scoped
# MAGIC   * Conda Runtime

# COMMAND ----------

# MAGIC %md
# MAGIC #### Databricks Runtime test1234
# MAGIC 
# MAGIC [Databricks Runtimes](https://docs.databricks.com/runtime/index.html#databricks-runtimes) are the set of core components that run on Databricks clusters.  These components include general [optimizations](https://docs.databricks.com/delta/optimizations/index.html#optimizations) to Delta Lake and Apache Spark, as well as pre-installed libraries for R, Python, and Scala/Java. In particular, the [Machine Learning Runtime](https://docs.databricks.com/runtime/mlruntime.html#mlruntime) has stable versions of XGBoost, Tensorflow, and Pytorch installed and configured to work out of the box.  These components help make Databricks a platform to develop quickly, with minimal library setup and debugging required. 
# MAGIC 
# MAGIC To learn more about what is in each runtime, it is worth reading the [release notes](https://docs.databricks.com/release-notes/runtime/supported.html#release-notes) 

# COMMAND ----------

# MAGIC %md 
# MAGIC #### test1234 In this demo, we are going to use Databricks ML Runtime to perform Airbnb Listing Data Analysis following the steps: Ingest &rarr; Explore &rarr; Enrich &rarr; Apply ML
# MAGIC 
# MAGIC <div style="width: 1100px;">
# MAGIC <img src="http://t2.gstatic.com/images?q=tbn:ANd9GcTotCbOIUt9xNehNqt4yAd8x19i3mo0Of_xccsc6V2KBh7j2W7B" width="8%" style="float: left; margin-right: 20px;">
# MAGIC     __Inside Airbnb Dataset__
# MAGIC   <div style="margin: 5 auto; width: 200px; height: 30px;"> <!-- use `background: green;` for testing -->
# MAGIC     <ul>
# MAGIC       <li> [ Dataset Reference - Inside Airbnb](http://insideairbnb.com/get-the-data.html)</li>
# MAGIC     </ul>
# MAGIC   </div>
# MAGIC </div>
# MAGIC <p>    </p>    

# COMMAND ----------

# MAGIC %md ###1. Ingest the Dataset test1234

# COMMAND ----------

dbutils.widgets.text("Search Parallelism", "8")
dbutils.widgets.text("Experiment Name", "/Users/Layla.Yang@databricks.com/Knowledge/ML-DL/Single Node Science Exp")

# COMMAND ----------

import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("/dbfs/mnt/training/airbnb/sf-listings/airbnb-cleaned-mlflow.csv")

df.head(5)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Let's Install `plotly` for High Quality Visualization for EDA.  test1234
# MAGIC 
# MAGIC *Note: We can install plotly 'inline' using the Conda environment available on the driver node. Simply use the `sh` syntax to install the library and it will be available in the next cell.*
# MAGIC 
# MAGIC <p></p>
# MAGIC 
# MAGIC 
# MAGIC <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSI1Gnw3Y4cf7BS8RNxxucajR5X9q0Zikj74AnUtSE448YGIVe_&s",  width =300>

# COMMAND ----------

# MAGIC %sh
# MAGIC conda install plotly

# COMMAND ----------

import plotly.express as px

fig = px.scatter(df, x='neighbourhood_cleansed', y='price'
                 ,size='accommodates'
                 , hover_data=['bedrooms', 'bathrooms', 'property_type', 'number_of_reviews']
                 ,color= 'room_type')
fig.update_layout(template='plotly_white')
fig.update_layout(title='How much should you charge per neighborhood?')
fig.show()

# COMMAND ----------

# MAGIC %md ###2. Explore the Dataset Using SQL Table - Access all of your data with Spark

# COMMAND ----------

sdf = spark.createDataFrame(df) 

sdf.write.mode("overwrite").saveAsTable("airbnb.listing")

# COMMAND ----------

# DBTITLE 1,Scatter plots to Illustrate Correlations Which May Impact Our Model
display(sdf)

# COMMAND ----------

# DBTITLE 1,What does the `price` distribution look like?
# MAGIC %sql
# MAGIC select price from airbnb.listing

# COMMAND ----------

display(sdf.select("price").describe())

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### 3. Use Pandas Dataframe To Create `matplotlib` Plots 
# MAGIC 
# MAGIC ##### Interoperability Between SQL Tables and Pandas Data Frames
# MAGIC 
# MAGIC Switching from a SQL Table or Spark Data Frame is Easy. Here we'll convert our data back to a pandas data frame using the toPandas() syntax. But if you prefer to run Pandas on Spark (for larger datasets), you can check out [Koalas](https://databricks.com/blog/2019/04/24/koalas-easy-transition-from-pandas-to-apache-spark.html).
# MAGIC 
# MAGIC <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRnDIrTYsjsK0ggYb5UiIrCEzWPDSGiJoSyDcryQUARjgnMepigPA&s", width=400>

# COMMAND ----------

def transfPlot(df, var, plot_name=None):
  '''
  two plots: one original var value
  one log transform value
  '''
  if plot_name is None:
      plot_name = ''

#   %matplotlib inline
  import pylab as pl
  import seaborn as sns
  import matplotlib.pyplot as plt
  from scipy import stats
  sns.set(color_codes=True)

  df_no_outliers = df[ (np.abs(df[var]-df[var].mean())<=(2*df[var].std()))&\
                (df[var] > 0)] # be careful there are high volumn of zeros in the dataset 

  print(df_no_outliers.shape)

  fig,axes = plt.subplots(ncols=2,nrows=2)
  fig.set_size_inches(12, 10)

  sns.distplot(df[var],ax=axes[0][0])
  stats.probplot(df[var], dist='norm', sparams=(2.5,),fit=True, plot=axes[0][1])

  # log transform
  sns.distplot(np.log(df_no_outliers[var]),ax=axes[1][0])
  stats.probplot(np.log1p(df_no_outliers[var]), dist='norm',sparams=(2.5,), fit=True, plot=axes[1][1])

  display(fig)
   

def plotFeatures(df, featureList, plot_name=None):
  if plot_name is None:
      plot_name = ''
  for var in featureList:
      var = var
      try:
          print("\n", "plot for: ", var)
          transfPlot(df, var, plot_name)
      except:
          print("\n", var, "Not a good feature!\n")


# COMMAND ----------

import numpy as np

pdf = sdf.toPandas()
pdf['log_price'] = np.log(pdf['price'])

df_plot = pdf
col_list = ['price','number_of_reviews']
plotFeatures(df_plot, col_list)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ###4. Scale Up Model Selection with Native AutoML Search Capabilities and MLflow
# MAGIC 
# MAGIC For models with long training times, start experimenting with small datasets and as many hyperparameters as possible. Use MLflow to introspect the best performing models, make informed decisions about how to fix as many hyperparameters as you can, and intelligently down-scope the parameter space as you prepare for tuning at scale.
# MAGIC 
# MAGIC <img src= "https://res.infoq.com/presentations/mlflow-databricks/en/slides/sl7-1566324281154.jpg", width = 500>

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ##### Prereqs - Set up MLflow experiment (place in the same folder for convenience) and train/test data

# COMMAND ----------

import mlflow
mlflow.set_experiment(dbutils.widgets.get("Experiment Name"))
experiment_name = dbutils.widgets.get("Experiment Name")

pdf = sdf.toPandas()
pdf['log_price'] = np.log(pdf['price'])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(pdf, pdf['log_price'], test_size=0.33, random_state=42)

# COMMAND ----------

from hyperopt import fmin, tpe, rand, hp, Trials, STATUS_OK
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
import mlflow

def train(params):
  """
  An example train method that computes the square of the input.
  This method will be passed to `hyperopt.fmin()`.
  
  :param params: hyperparameters. Its structure is consistent with how search space is defined. See below.
  :return: dict with fields 'loss' (scalar loss) and 'status' (success/failure status of run)
  """
  curr_model =  XGBRegressor(learning_rate=params[0],
                            gamma=int(params[1]),
                            max_depth=int(params[2]),
                            n_estimators=int(params[3]),
                            min_child_weight = params[4], objective='reg:linear')
  score = -cross_val_score(curr_model, X_train, y_train, scoring='neg_mean_squared_error').mean()
  score = np.array(score)
  
  return {'loss': score, 'status': STATUS_OK, 'model': curr_model}

# COMMAND ----------

# define search parameters and whether discrete or continuous
search_space = [ hp.uniform('learning_rate', 0, 1),
                 hp.uniform('gamma', 0, 5),
                 hp.quniform('max_depth', 0, 10, 1),
                 hp.quniform('n_estimators', 0, 20, 1),
                 hp.quniform('min_child_weight', 0, 10, 1)
               ]
# define the search algorithm (TPE or Randomized Search)
# can choose tpe for more complex search but it requires more trials
# MORE COMMENT
algo= tpe.suggest

# COMMAND ----------

from hyperopt import SparkTrials

search_parallelism = int(dbutils.widgets.get("Search Parallelism"))
spark_trials = SparkTrials(parallelism=search_parallelism)

with mlflow.start_run():
  argmin = fmin(
    fn=train,
    space=search_space,
    algo=algo,
    max_evals=8,
    trials=spark_trials)

# COMMAND ----------

def fit_best_model(X, y): 
  client = mlflow.tracking.MlflowClient()
  experiment_id = client.get_experiment_by_name(experiment_name).experiment_id

  runs = mlflow.search_runs(experiment_id)
  best_loss = runs['metrics.loss'].min()
  best_run=runs[runs['metrics.loss'] == best_loss]

  best_params = {}
  best_params['gamma'] = float(best_run['params.gamma'])
  best_params['learning_rate'] = float(best_run['params.learning_rate'])
  best_params['max_depth'] = float(best_run['params.max_depth'])
  best_params['min_child_weight'] = float(best_run['params.min_child_weight'])  
  best_params['n_estimators'] = float(best_run['params.n_estimators'])
  
  xgb_regressor =  XGBRegressor(learning_rate=best_params['learning_rate'],
                            max_depth=int(best_params['max_depth']),
                            n_estimators=int(best_params['n_estimators']),
                            gamma=int(best_params['gamma']),
                            min_child_weight = best_params['min_child_weight'], objective='reg:linear')

  xgb_model = xgb_regressor.fit(X, y)
  return(xgb_model)

# COMMAND ----------

X = pdf.drop(["price", "log_price"], axis=1)
y = pdf["price"]

xgb_model = fit_best_model(X, y)

# COMMAND ----------

import shap

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X, y=y.values)
mean_abs_shap = np.absolute(shap_values).mean(axis=0).tolist()
#need to create spark dataframe to use display
display(spark.createDataFrame(sorted(list(zip(mean_abs_shap, X.columns)), reverse=True)[:8], ["Mean |SHAP|", "Column"]))

# COMMAND ----------

# DBTITLE 1,Explain an individual listing using features
pdf[10:11]

# COMMAND ----------

plot_html = shap.force_plot(explainer.expected_value, shap_values[10:11], feature_names=X.columns, plot_cmap='GnPR')
displayHTML(bundle_js + plot_html.data)
