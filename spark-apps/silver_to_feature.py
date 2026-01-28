from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

spark = SparkSession.builder.appName("FeatureEngineering").config("spark.driver.memory", "8g").config("spark.executor.memory", "8g").getOrCreate()

df = spark.read.parquet("spark-data/silver/transaction_clean")

w_orig = Window.partitionBy("nameOrig").orderBy("step")
w_orig_past = w_orig.rowsBetween(Window.unboundedPreceding, -1)
w_orig_24h_past = w_orig.rangeBetween(-24, -1)

w_dest = Window.partitionBy("nameDest").orderBy("step")
w_dest_past = w_dest.rowsBetween(Window.unboundedPreceding, -1)
w_dest_1h_past = w_dest.rangeBetween(-1, -1)
w_dest_24h_past = w_dest.rangeBetween(-24, -1)

q1, q3 = df.approxQuantile("amount", [0.25, 0.75], 0.01)
iqr = q3 - q1
LOWER_BOUND = q1 - 1.5 * iqr
UPPER_BOUND = q3 + 1.5 * iqr

features_df = (
df
# -------- Balance Consistency --------
.withColumn("errorBalanceOrig", (F.col("oldbalanceOrg") - F.col("amount")) - F.col("newbalanceOrig"))
.withColumn("errorBalanceDest", (F.col("oldbalanceDest") + F.col("amount")) - F.col("newbalanceDest"))


.withColumn("is_errorBalanceOrig", F.when(F.col("errorBalanceOrig") != 0, 1).otherwise(0))
.withColumn("is_errorBalanceDest", F.when(F.col("errorBalanceDest") != 0, 1).otherwise(0))


# -------- Transaction Type --------
.withColumn("is_transfer", F.when(F.col("type") == "TRANSFER", 1).otherwise(0))
.withColumn("is_cash_out", F.when(F.col("type") == "CASH_OUT", 1).otherwise(0))


# -------- Time Context --------
.withColumn("hour_of_day", F.col("step") % 24)
.withColumn("day_of_week", (F.col("step") / 24).cast("int") % 7)


# -------- Temporal Delta (Origin) --------
.withColumn("orig_prev_step", F.lag("step").over(w_orig))
.withColumn("orig_delta_step", F.col("step") - F.col("orig_prev_step"))


# -------- Temporal Delta (Destination) --------
.withColumn("dest_prev_step", F.lag("step").over(w_dest))
.withColumn("dest_delta_step", F.col("step") - F.col("dest_prev_step"))


# -------- Destination Historical Aggregates --------
.withColumn("dest_count_past", F.count("*").over(w_dest_past))
.withColumn("dest_sum_amount_past", F.sum("amount").over(w_dest_past))
.withColumn("dest_hist_fraud", F.sum("isFraud").over(w_dest_past))


# -------- Destination Short-term Bursts --------
.withColumn("dest_tx_count_1h", F.count("*").over(w_dest_1h_past))
.withColumn("dest_amount_sum_1h", F.sum("amount").over(w_dest_1h_past))


.withColumn("dest_tx_count_24h", F.count("*").over(w_dest_24h_past))
.withColumn("dest_amount_sum_24h", F.sum("amount").over(w_dest_24h_past))


# -------- Account State Flags --------
.withColumn("is_all_orig_balance", F.when(F.col("amount") == F.col("oldbalanceOrg"), 1).otherwise(0))
.withColumn("is_dest_zero_init", F.when(F.col("oldbalanceDest") == 0, 1).otherwise(0))
.withColumn("is_org_zero_init", F.when(F.col("oldbalanceOrg") == 0, 1).otherwise(0))


# -------- Amount Outlier --------
.withColumn("is_amount_outlier", F.when((F.col("amount") < LOWER_BOUND) | (F.col("amount") > UPPER_BOUND), 1).otherwise(0)))


features_df = features_df.fillna(0)
features_df.write.mode("overwrite").parquet("spark-data/feature/transaction_features")
print("Feature engineering completed successfully")

spark.stop()
print("Spark Session has stopped.")