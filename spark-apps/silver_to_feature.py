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

    df_features = df.withColumn("errorBalanceOrig", (col("oldbalanceOrg") - col("amount")) - col("newbalanceOrig")) \
                    .withColumn("errorBalanceDest", (col("oldbalanceDest") + col("amount")) - col("newbalanceDest"))

    # 2. Mã hóa loại giao dịch (One-hot encoding đơn giản bằng 'when')
    df_features = df_features.withColumn("is_transfer", when(col("type") == "TRANSFER", 1).otherwise(0)) \
                             .withColumn("is_cash_out", when(col("type") == "CASH_OUT", 1).otherwise(0))

    # 3. Tạo đặc trưng 'người nhận là Merchant' (thường bắt đầu bằng chữ M)
    df_features = df_features.withColumn("is_merchant_dest", when(col("merchant_id").startsWith("M"), 1).otherwise(0))

    # 4. Ghi dữ liệu ra tầng Feature/Gold
    df_features.coalesce(1).write.mode("overwrite").parquet("spark-data/feature/transaction_features")

finally:
    spark.stop()
    print("Spark Session has stopped.")