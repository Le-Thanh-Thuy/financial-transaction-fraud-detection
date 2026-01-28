from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp

spark = SparkSession.builder.appName("TransactionBronze").config("spark.driver.memory", "4g").config("spark.executor.memory", "4g").getOrCreate()

df = spark.read.option("header", True).csv("spark-data/source/PS_20174392719_1491204439457_log.csv") 
df = df.withColumn("ingest_time", current_timestamp())
df.write.mode("overwrite").parquet("spark-data/bronze/transaction")
print("Data ingestion successful!")

spark.stop()
print("Spark Session has stopped.")