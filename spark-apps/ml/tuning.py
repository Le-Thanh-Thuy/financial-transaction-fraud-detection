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
    
    # Lấy thông số tốt nhất
    best_pipeline_model = cv_model.bestModel
    # Model nằm ở stage cuối cùng của pipeline
    best_lr_model = best_pipeline_model.stages[-1]
    
    print(f"--- BEST PARAMS ---")
    print(f"RegParam: {best_lr_model._java_obj.getRegParam()}")
    print(f"ElasticNetParam: {best_lr_model._java_obj.getElasticNetParam()}")
    
    return best_pipeline_model