# ml/tuning.py
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline

def tune_best_model(train_df, stages, model_obj):
    # Lưới tham số cho Logistic Regression
    paramGrid = ParamGridBuilder() \
        .addGrid(model_obj.regParam, [0.01, 0.1, 1.0]) \
        .addGrid(model_obj.elasticNetParam, [0.0, 0.5, 1.0]) \
        .build()

    evaluator = BinaryClassificationEvaluator(labelCol="isFraud", metricName="areaUnderROC")

    pipeline = Pipeline(stages=stages + [model_obj])
    # Sử dụng 3-fold để cân bằng giữa độ chính xác và tài nguyên
    cv = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=paramGrid,
        evaluator=evaluator,
        numFolds=3,
        parallelism=2
    ) 

    cv_model = cv.fit(train_df)
    print(f"Tuning hoàn tất.")
    return cv_model.bestModel