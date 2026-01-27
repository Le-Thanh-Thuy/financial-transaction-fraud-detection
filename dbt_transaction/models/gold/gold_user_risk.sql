select
    user_id,
    txn_count,
    avg_amount,
    fraud_count,
    case
        when txn_count > 20 and avg_amount > 500 then 1
        else 0
    end as rule_risk_flag,
    case
        when fraud_count > 0 then 1
        else 0
    end as label
from {{ ref('stg_user_features') }}