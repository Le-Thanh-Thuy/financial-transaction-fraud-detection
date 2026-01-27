from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp

spark = SparkSession.builder.appName("TransactionBronze").config("spark.driver.memory", "4g").config("spark.executor.memory", "4g").getOrCreate()

# spark = SparkSession.builder \
#     .appName("Transaction-Bronze") \
#     .master("spark://spark-master:7077") \
#     .config("spark.executor.instances", "3") \
#     .config("spark.executor.cores", "2") \
#     .config("spark.executor.memory", "2g") \
#     .getOrCreate()

try:
    df = spark.read.option("header", True).csv("spark-data/source/PS_20174392719_1491204439457_log.csv") 
    # df = spark.read.option("header", True).csv("/opt/spark-data/source/PS_20174392719_1491204439457_log.csv")

    df = df.withColumn("ingest_time", current_timestamp())

    df.coalesce(1).write.mode("overwrite").parquet("spark-data/bronze/transaction")

    print("Data ingestion successful!")

finally:
    spark.stop()
    print("Spark Session has stopped.")