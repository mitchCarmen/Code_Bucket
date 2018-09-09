################ PySpark is the Python API for Spark

# Good Resource:
# http://spark.apache.org/docs/2.1.0/api/python/pyspark.html

# Splitting data and compute power across many clusters and nodes speeds up everything
# 'Master' controls everything : slaves
# Spark's core data structure is an RDD -- Resilient Distributed Dataset
# Work with Spark DataFrames rather than RDDs! It's just easier...


# START SPARK FROM LOCAL MACHINE
# Normalizing Data
# Standardizing Data
# Bucketize Numeric Data


#####################################################
#################### START SPARK FROM LOCAL MACHINE

/Users/Mitch/Documents/spark/bin
Mitchells-MacBook-Air-3:bin Mitch$ ./pyspark

# ctrl + L clears the screen

# Reading CSV files
df = spark.read.csv("/Users/Mitch/Documents/spark/bin/employee.txt", header = True)

df.schema # Shows schema

df.printSchema() # More readable

df.columns # Prints columns

df.take(5) # lists first 5 rows of df

df.count() # tells number of rows

sample_df = df.sample(False, 0.1) # False for sampling without replacement, and taking 10% of the data
# Great for working with very large dataset

df_smaller = df.filter("salary >= 100000") # another good way to downsize date by sampling

df_smaller.select("salary").show() # Lists top 20 items


# Normalizing Data
from pyspark.ml.feature import MinMaxScaler
from pyspar.ml.linalg import Vectors

features_df = spark.createDataFrame([
	(1, Vectors.dense([10.0, 10000.0, 1.0]),),
	(2, Vectors.dense([20.0, 30000.0, 2.0]),),
	(3, Vectors.dense([30.0, 40000.0, 3.0]),)
	], ["id","features"])
features_df.take(1) # To show first item

feature_scaler = MinMaxScaler(inputCol="features",outputCol="sfeatures")
smodel = feature_scaler.fit(features_df)
scaled_features_df = smodel.transform(features_df)

scaled_features_df.take(1)

scaled_features_df.select("features","scaled_features_df").show() # shows columns of orig and scaled features


# Standardizing Data
# Helps to transform data to normal distributions for GLM and SVMs
from pyspark.ml.feature import StandardScaler
from pyspar.ml.linalg import Vectors

features_df = spark.createDataFrame([
	(1, Vectors.dense([10.0, 10000.0, 1.0]),),
	(2, Vectors.dense([20.0, 30000.0, 2.0]),),
	(3, Vectors.dense([30.0, 40000.0, 3.0]),)
	],["id","features"])
features_df.take(1) # To show first row

feature_stand_scaler = StandardScaler(inputCol="features",outputCol="sfeatures",withStd=True,withMean=True)
stand_smodel = feature_stand_scaler.fit(features_df)
stand_sfeatures_df = stand_smodel.transform(features_df)

stand_sfeatures_df.take(1) #take a look
stand_sfeatures_df.show() 


# Bucketize Numeric Data
from pyspark.ml.feature import Bucketizer
splits = [-float("inf"), -10.0, 0.0, 10.0, float("inf")]

# Creating some data
b_data = list of big data

b_df = spark.createDataFrame(b_data, ["features"])
b_df.show()

bucketizer = Bucketizer(splits=splits, inputCol="features", outputCol="bfeatures")
bucketed_df = bucketizer.transform(b_df) # Bucketizer does not need to be fit

bucketed_df.show() # They get assigned values of their buckets 










#####################################################
#################### SET UP CONNECTION

# 1: Set up Spark Connection to cluster
sc = SparkContext(conf=conf)

# 2: Set up Spark Session to interface into the connection
from pyspark.sql import SparkSession
my_spark = SparkSession.builder.getOrCreate()


#####################################################
#################### LOOKING AT DATA

# Print the tables in the catalog
print(spark.catalog.listTables())

### RUN A QUERY
# Don't change this query
query = "FROM flights SELECT * LIMIT 10"
# Get the first 10 rows of flights
flights10 = spark.sql(query)
# Show the results
flights10.show()

### CHANGE TO PANDAS DF FROM SPARK CLUSTER
# Don't change this query
query = "SELECT origin, dest, COUNT(*) as N FROM flights GROUP BY origin, dest"
# Run the query
flight_counts = spark.sql(query)
# Convert the results to a pandas DataFrame
pd_counts = flight_counts.toPandas()
# Print the head of pd_counts
print(pd_counts.head())
