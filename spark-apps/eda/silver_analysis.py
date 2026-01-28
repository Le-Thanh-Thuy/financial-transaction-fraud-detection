import json
import warnings
from pathlib import Path
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, avg, sum, min, max, stddev, percentile_approx
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
from pyspark.sql import functions as F
warnings.filterwarnings('ignore')

CONFIG = {
    'app_name': 'Fraud-EDA-Spark-Only',
    'silver_path': 'spark-data/silver/transaction_clean',
    'output_dirs': {
        'reports': 'spark-data/silver-eda-reports',
    },
    'target_col': 'isFraud'
}

Path(CONFIG['output_dirs']['reports']).mkdir(parents=True, exist_ok=True)

def print_header(title):
    print("\n" + "="*80)
    print(f" {title.upper()}")
    print("="*80)

def get_spark_session():
    return SparkSession.builder \
        .appName(CONFIG['app_name']) \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

        # .config("spark.sql.shuffle.partitions", "100") \
def compute_and_save_general_stats(df):
    print_header("General Dataset Overview")
    
    total_rows = df.count()
    total_cols = len(df.columns)
    print(f"Dataset shape: {total_rows:,} rows × {total_cols} columns")

    print("\nData types:")
    dtype_groups = {}
    for _, dtype in df.dtypes:
        dtype_groups[dtype] = dtype_groups.get(dtype, 0) + 1
    for dtype, n in dtype_groups.items():
        print(f"  {dtype}: {n} columns")
    
    stats = df.select(
        count("*").alias("total_rows"),
        sum(col(CONFIG['target_col'])).alias("fraud_rows"),
        F.concat(F.round((F.sum(col(CONFIG['target_col'])) / count("*") * 100), 3).cast("string"), F.lit(" %")).alias("fraud_rate_percentage"),
        F.round(avg("amount"),3).alias("avg_amount"),
        min("amount").alias("min_amount"),
        max("amount").alias("max_amount"),
        F.round(stddev("amount"),3).alias("std_amount")
    )
    
    print("\nOverview statistic:")
    stats.show(truncate=False)
    
    stats.coalesce(1).write.mode("overwrite") \
        .option("header", "true") \
        .csv(f"{CONFIG['output_dirs']['reports']}/general_stats")

    quality_report = {
        "dataset_shape": [total_rows, total_cols],
        "data_types": dtype_groups,
        'fraud_percentage': float((stats.collect()[0]['fraud_rows'] / stats.collect()[0]['total_rows']) * 100),
    }
    
    return quality_report 

def compute_and_save_type_stats(df):
    print_header("Fraud Analysis by Transaction Type")
    type_stats = df.groupBy("type").agg(
        count("*").alias("total_count"),
        sum(CONFIG['target_col']).alias("fraud_count"),
        avg("amount").alias("mean_amount")
    ).orderBy(col("fraud_count").desc())
    
    type_stats.show()
    
    type_stats.coalesce(1).write.mode("overwrite") \
        .option("header", "true") \
        .csv(f"{CONFIG['output_dirs']['reports']}/type_stats")

def compute_and_save_value_behavior(df):
    print_header("Value Behavior Analysis")
    
    value_stats = df.groupBy(CONFIG['target_col']).agg(
        count("*").alias("count"),
        min("amount").alias("min_amount"),
        percentile_approx("amount", 0.25).alias("q1_25pct"),
        percentile_approx("amount", 0.50).alias("median_50pct"),
        percentile_approx("amount", 0.75).alias("q3_75pct"),
        max("amount").alias("max_amount"),
        avg("amount").alias("mean_amount"),
        stddev("amount").alias("std_dev")
    )
    
    value_stats.show(truncate=False)
    
    value_stats.coalesce(1).write.mode("overwrite") \
        .option("header", "true") \
        .csv(f"{CONFIG['output_dirs']['reports']}/value_behavior_stats")

def compute_and_save_time_behavior(df):
    print_header("Time-based Behavior Analysis (Hour & DayOfWeek)")
    
    # 1. Trích xuất Hour và Day từ step
    # Giả định: step 1 = giờ đầu tiên của ngày đầu tiên
    time_df = df.filter(F.col(CONFIG['target_col']) == 1) \
        .withColumn("hour_of_day", F.col("step") % 24) \
        .withColumn("day_of_week", (F.col("step") / 24).cast("int") % 7)

    # 2. Thống kê theo Hour of Day (Giờ nào hay bị gian lận nhất?)
    print("\n[1] Fraud Stats by Hour of Day (0-23):")
    hour_stats = time_df.groupBy("hour_of_day").agg(
        F.count("*").alias("fraud_count"),
        F.round(F.sum("amount"), 2).alias("total_fraud_amount"),
        F.round(F.avg("amount"), 2).alias("avg_fraud_amount")
    ).orderBy(F.col("fraud_count").desc())
    hour_stats.show(24)

    # 3. Thống kê theo Day of Week (Thứ mấy hay bị gian lận nhất?)
    print("\n[2] Fraud Stats by Day of Week:")
    day_stats = time_df.groupBy("day_of_week").agg(
        F.count("*").alias("fraud_count"),
        F.round(F.sum("amount"), 2).alias("total_fraud_amount")
    ).orderBy(F.col("day_of_week").asc())
    day_stats.show()

    # 4. Lưu báo cáo
    hour_stats.coalesce(1).write.mode("overwrite").option("header", "true") \
        .csv(f"{CONFIG['output_dirs']['reports']}/fraud_by_hour")
    
    day_stats.coalesce(1).write.mode("overwrite").option("header", "true") \
        .csv(f"{CONFIG['output_dirs']['reports']}/fraud_by_day")

    print(f"Time-based reports saved to {CONFIG['output_dirs']['reports']}")

def compute_null_metrics(df):
    print_header("Data Quality Metrics (Null Counts)")
    null_exprs = [sum(col(c).isNull().cast("int")).alias(c) for c in df.columns]
    null_df = df.select(null_exprs)
    
    null_df.show()
    
    null_df.coalesce(1).write.mode("overwrite") \
        .option("header", "true") \
        .csv(f"{CONFIG['output_dirs']['reports']}/null_metrics")

def compute_and_save_correlation_matrix(df):
    print_header("Feature Correlation Matrix")
    numeric_cols = [c for c, t in df.dtypes if t in ('double', 'int', 'float', 'long')]
    cols_to_corr = [c for c in numeric_cols if c != 'isFlaggedFraud'] 
    assembler = VectorAssembler(inputCols=cols_to_corr, outputCol="features", handleInvalid="skip")
    df_vector = assembler.transform(df).select("features")
    matrix = Correlation.corr(df_vector, "features").collect()[0][0]
    corr_matrix = matrix.toArray().tolist()

    corr_data = []
    for i in range(len(cols_to_corr)):
        for j in range(len(cols_to_corr)):
            corr_data.append((cols_to_corr[i], cols_to_corr[j], float(corr_matrix[i][j])))
    
    corr_df = spark.createDataFrame(corr_data, ["feature_1", "feature_2", "correlation"])

    print(f"Correlation with Target Variable {CONFIG['target_col']}:")
    corr_df.filter(col("feature_2") == CONFIG['target_col']).orderBy(col("correlation").desc()).show()

    # 5. Lưu kết quả
    corr_df.coalesce(1).write.mode("overwrite") \
        .option("header", "true") \
        .csv(f"{CONFIG['output_dirs']['reports']}/correlation_matrix")

def analyze_column_details(df):
    print_header("Column Data Quality Details")
    agg_exprs = []
    for c, _ in df.dtypes:
        agg_exprs.append(F.sum(F.col(c).isNull().cast("int")).alias(f"{c}_null"))
        agg_exprs.append(F.count_distinct(F.col(c)).alias(f"{c}_unique"))
    metrics_row = df.select(agg_exprs).collect()[0].asDict()
    print("\nDetailed Metrics per Column:")
    print("-" * 60)
    
    dtype_map = {"double": "float64", "bigint": "int64", "int": "int32", "string": "object"}

    for c, dtype in df.dtypes:
        null_count = metrics_row[f"{c}_null"]
        unique_count = metrics_row[f"{c}_unique"]
        
        display_dtype = dtype_map.get(dtype, dtype)
        
        print(f" {c} ({dtype}, {unique_count:,} unique, {null_count:,} missing)")
        
    print("-" * 60)

def compute_and_save_outliers(df):
    print_header("Outlier Detection (IQR Method)")
    
    numeric_cols = [c for c, t in df.dtypes if t in ('double', 'int', 'float', 'long')]
    outlier_report = []

    print(f"Analyzing outliers for {len(numeric_cols)} columns...")

    for c in numeric_cols:
        # 1. Tính Q1 và Q3
        quants = df.select(
            F.percentile_approx(c, 0.25).alias("q1"),
            F.percentile_approx(c, 0.75).alias("q3")
        ).collect()[0]
        
        # Đảm bảo giá trị không bị None (tránh lỗi tính toán)
        q1 = float(quants["q1"]) if quants["q1"] is not None else 0.0
        q3 = float(quants["q3"]) if quants["q3"] is not None else 0.0
        iqr = q3 - q1
        
        # 2. Xác định ngưỡng Bound
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # 3. Đếm số lượng Outliers
        outliers_count = df.filter((F.col(c) < lower_bound) | (F.col(c) > upper_bound)).count()
        total_count = df.count()
        percentage = (outliers_count / total_count) * 100 if total_count > 0 else 0.0
        
        print(f"  {c:20s}: {outliers_count:10,} outliers ({percentage:.2f}%)")
        
        # 4. ÉP KIỂU TẤT CẢ VỀ FLOAT TRƯỚC KHI APPEND
        # Việc ép kiểu thủ công ở đây giúp Spark tạo Schema đồng nhất (DoubleType)
        outlier_report.append((
            str(c), 
            float(q1), 
            float(q3), 
            float(iqr), 
            float(lower_bound), 
            float(upper_bound), 
            float(outliers_count), # Chuyển count sang float để đồng nhất type
            float(percentage)
        ))

    # 5. Lưu kết quả ra CSV
    # Spark sẽ nhận diện toàn bộ là DoubleType, không còn lỗi Merge Type
    outlier_df = spark.createDataFrame(outlier_report, 
        ["column", "q1", "q3", "iqr", "lower_bound", "upper_bound", "outlier_count", "percentage"])
    
    outlier_df.coalesce(1).write.mode("overwrite") \
        .option("header", "true") \
        .csv(f"{CONFIG['output_dirs']['reports']}/outlier_analysis")
    
    print(f"Outlier analysis saved to {CONFIG['output_dirs']['reports']}/outlier_analysis")

# =============================================================================
# MAIN PIPELINE
# =============================================================================
if __name__ == "__main__":
    start_time = datetime.now()
    spark = get_spark_session()
    
    try:
        print_header("Starting Spark-Only EDA Pipeline")
        
        # 1. Load Data
        df = spark.read.parquet(CONFIG['silver_path'])
        # df.cache()

        # 2. Xử lý & Lưu trữ (Hoàn toàn bằng Spark)
        report = compute_and_save_general_stats(df)
        compute_null_metrics(df)
        compute_and_save_outliers(df)
        compute_and_save_type_stats(df)
        compute_and_save_value_behavior(df)
        compute_and_save_time_behavior(df)
        compute_and_save_correlation_matrix(df)
        
        # 3. Xuất báo cáo tóm tắt JSON
        report['execution_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report['processing_time_sec'] = (datetime.now() - start_time).total_seconds()
        
        with open(Path(CONFIG['output_dirs']['reports']) / 'eda_summary.json', 'w') as f:
            json.dump(report, f, indent=4)
            
        print_header("EDA Pipeline Completed Successfully")
        print(f"Total Time: {report['processing_time_sec']:.2f} seconds")

    except Exception as e:
        print(f"Pipeline Failed: {str(e)}")
    finally:
        spark.stop()