# ml/utils.py
from pyspark.ml.feature import VectorAssembler, QuantileDiscretizer
from pyspark.sql.functions import when, col, lit
from pyspark.sql import functions as F

def prepare_pipeline_data(df, target_col):
    feature_cols = [
        'amount', 

        'is_transfer', 
        'is_cash_out', 
        'hour_of_day',
        'day_of_week',
        
        'orig_prev_step', 
        'orig_delta_step', 
        'dest_prev_step', 
        'dest_delta_step', 
        
        'dest_tx_count_1h', 
        'dest_amount_sum_1h', 

        'is_dest_zero_init', 
        'is_org_zero_init', 
        # 'is_amount_outlier'
    ]
    
    # Tính toán trọng số cân bằng nhãn (Class Weighting)
    counts = df.groupBy(target_col).count().collect()
    count_dict = {row[target_col]: row['count'] for row in counts}
    fraud_count = count_dict.get(1, 0)
    non_fraud_count = count_dict.get(0, 1)
    
    ratio = non_fraud_count / fraud_count if fraud_count > 0 else 1.0
    # Gán trọng số cao hơn cho lớp Fraud (1.0) để cải thiện Recall
    df = df.withColumn("weight", when(col(target_col) == 1, ratio).otherwise(1.0))
    
    return df, feature_cols

def get_discretizer(feature_cols):
    output_cols = [c + "_bin" for c in feature_cols]
    discretizer = QuantileDiscretizer(
        numBuckets=10, 
        inputCols=feature_cols, 
        outputCols=output_cols,
        handleInvalid="keep",
        relativeError=0.01
    )
    return discretizer, output_cols

def split_data_by_time(df, train_ratio):
    # Chia dữ liệu theo trình tự thời gian dựa trên step
    threshold_dict = df.stat.approxQuantile("step", [train_ratio], 0.01)
    threshold_step = threshold_dict[0]
    train_df = df.filter(F.col("step") <= threshold_step)
    test_df = df.filter(F.col("step") > threshold_step)    
    return train_df, test_df