from config import *
from utils import prepare_pipeline_data, get_discretizer, split_data_by_time
from tuning import tune_best_model
from metrics import evaluate_spark_model

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler, ChiSqSelector
from pyspark.sql.functions import approx_count_distinct, col
from pyspark.sql import functions as F, SparkSession
import gc, os

def main():
    spark = SparkSession.builder \
        .appName("FraudDetectionTraining") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()
    
    # 1. Load Data & Split
    df = spark.read.parquet(DATA_PATH).filter(col("type").isin("CASH_OUT", "TRANSFER"))
    train_df, test_df = split_data_by_time(df, TRAIN_SIZE)
    
    # 2. Chuẩn bị đặc trưng (chỉ tính trên tập Train để tránh leakage)
    train_df, feature_cols = prepare_pipeline_data(train_df, TARGET_COL)
    test_df = test_df.withColumn("weight", F.lit(1.0)) # Test không dùng trọng số cân bằng

    stages = []
    final_features_col = "raw_features"

    continuous_cols, categorical_cols = [], []
    for c in feature_cols:
        distinct_count = train_df.select(approx_count_distinct(c)).collect()[0][0]
        if distinct_count > 100: 
            continuous_cols.append(c)
        else: 
            categorical_cols.append(c)

    assembler_cat = VectorAssembler(inputCols=categorical_cols, outputCol="cat_features")
    
    selector = ChiSqSelector(
        numTopFeatures=min(10, len(categorical_cols)), 
        featuresCol="cat_features", 
        outputCol="selected_cat_features", 
        labelCol=TARGET_COL
    )
    
    # Bước C: Gộp các biến rời rạc đã lọc VỚI các biến liên tục thô
    final_assembler = VectorAssembler(
        inputCols=["selected_cat_features"] + continuous_cols, 
        outputCol="final_features"
    )
    
    stages += [assembler_cat, selector, final_assembler]
    final_features_col = "final_features"

    # 3. Khởi tạo model Logistic Regression
    model_obj = LogisticRegression(labelCol="isFraud", featuresCol="raw_features", weightCol="weight")
    model_obj.setFeaturesCol(final_features_col)
    
    # print("--- Huấn luyện Model gốc ---")
    # initial_pipeline = Pipeline(stages=stages + [model_obj])
    # initial_model = initial_pipeline.fit(train_df)
    
    # 4. Tuning
    best_tuned_pipeline = tune_best_model(train_df, stages, model_obj)
    
    # 5. Đánh giá và lưu kết quả
    final_preds = best_tuned_pipeline.transform(test_df)
    final_res = evaluate_spark_model(final_preds)
    
    # Lưu bảng Metric vào CSV
    metrics_df = pd.DataFrame([final_res])
    header = not os.path.exists(METRIC_PATH)
    metrics_df.to_csv(METRIC_PATH, mode='a', index=False, header=header)
    
    # Lưu Model
    best_tuned_pipeline.write().overwrite().save(MODEL_PATH)
    
    print(pd.DataFrame([final_res]).to_markdown(index=False))
    spark.stop()

if __name__ == "__main__":
    main()