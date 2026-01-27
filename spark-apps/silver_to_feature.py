from pyspark.sql import SparkSession
from pyspark.sql.functions import count, sum, avg

spark = SparkSession.builder.appName("FeatureEngineering").config("spark.driver.memory", "4g").config("spark.executor.memory", "4g").getOrCreate()

try:
    df = spark.read.parquet("spark-data/silver/transaction_clean")

    user_features = df.groupBy("user_id").agg(
        count("*").alias("txn_count"),
        sum("amount").alias("total_amount"),
        avg("amount").alias("avg_amount"),
        sum("isFraud").alias("fraud_count")
    )

    user_features.coalesce(1).write.mode("overwrite").parquet("spark-data/feature/user_features")
    print("Data processing successful!")

finally:
    spark.stop()
    print("Spark Session has stopped.")