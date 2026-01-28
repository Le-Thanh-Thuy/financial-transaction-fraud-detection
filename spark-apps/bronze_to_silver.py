from pyspark.sql import SparkSession
from pyspark.sql.functions import col, trim, current_timestamp

spark = SparkSession.builder.appName("TransactionSilver").config("spark.driver.memory", "4g").config("spark.executor.memory", "4g").getOrCreate()

df = spark.read.parquet("spark-data/bronze/transaction")

df_clean = df.select(
    col("step").cast("int"),
    trim(col("type")).alias("type"),
    col("amount").cast("double"),
    trim(col("nameOrig")).alias("nameOrig"),
    col("oldbalanceOrg").cast("double"),
    col("newbalanceOrig").cast("double"),
    trim(col("nameDest")).alias("nameDest"),
    col("oldbalanceDest").cast("double"),
    col("newbalanceDest").cast("double"),
    col("isFraud").cast("int"),
    col("isFlaggedFraud").cast("int")
)

df_clean = df_clean.dropDuplicates()
df_clean.write.mode("overwrite").parquet("spark-data/silver/transaction_clean")
print("Data cleaning successful!")

spark.stop()
print("Spark Session has stopped.")
