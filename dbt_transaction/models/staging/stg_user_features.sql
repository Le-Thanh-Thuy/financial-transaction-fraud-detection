select
    user_id,
    txn_count,
    total_amount,
    avg_amount,
    fraud_count
from read_parquet('../spark-data/feature/user_features/*.parquet')