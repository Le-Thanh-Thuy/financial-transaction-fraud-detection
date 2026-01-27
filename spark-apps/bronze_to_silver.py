from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder.appName("TransactionSilver").config("spark.driver.memory", "4g").config("spark.executor.memory", "4g").getOrCreate()

try:
    df = spark.read.parquet("spark-data/bronze/transaction")

    df_clean = df.select(
        col("step").cast("int"),
        col("type"),
        col("amount").cast("double"),
        col("nameOrig").alias("user_id"),
        col("oldbalanceOrg").cast("double"),
        col("newbalanceOrig").cast("double"),
        col("nameDest").alias("merchant_id"),
        col("oldbalanceDest").cast("double"),
        col("newbalanceDest").cast("double"),
        col("isFraud").cast("int")
    )

    df_clean.coalesce(1).write.mode("overwrite").parquet("spark-data/silver/transaction_clean")

    print("Data cleaning successful!")

finally:
    spark.stop()
    print("Spark Session has stopped.")
