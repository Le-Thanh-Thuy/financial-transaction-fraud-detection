import pandas as pd
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql.functions import col

def evaluate_spark_model(predictions):
    eval_auc = BinaryClassificationEvaluator(labelCol="isFraud", metricName="areaUnderROC")
    auc = eval_auc.evaluate(predictions)
    
    # Chuyển đổi sang RDD để dùng MulticlassMetrics tính Macro Metrics
    prediction_and_label = predictions.select(
        col("prediction").cast("double"), 
        col("isFraud").cast("double")
    ).rdd
    metrics = MulticlassMetrics(prediction_and_label)
    
    labels = [0.0, 1.0] # 0: Normal, 1: Fraud
    class_metrics = {}
    for label in labels:
        class_metrics[f"Prec_{int(label)}"] = metrics.precision(label)
        class_metrics[f"Rec_{int(label)}"] = metrics.recall(label)
        class_metrics[f"F1_{int(label)}"] = metrics.fMeasure(label, 1.0)

    # Macro Recall: Trung bình không trọng số của Recall từng lớp
    macro_recall = sum([metrics.recall(l) for l in labels]) / len(labels)
    predictions.groupBy("isFraud", "prediction").count().sort('isFraud').show()
    
    print(f"Accuracy : {round(metrics.accuracy, 6)}")
    print(f"Prec_Normal : {round(class_metrics['Prec_0'], 6)}")
    print(f"Rec_Normal : {round(class_metrics['Rec_0'], 6)}")
    print(f"Prec_Fraud : {round(class_metrics['Prec_1'], 6)}")
    print(f"Rec_Fraud : {round(class_metrics['Rec_1'], 6)}")
    print(f"Macro_Recall : {round(macro_recall, 6)}")
    print(f"F1_Weighted : {round(metrics.weightedFMeasure(), 6)}")
    print(f"AUC : {round(auc, 6)}")

    results = {
        "Accuracy": metrics.accuracy,
        "Prec_Normal": class_metrics['Prec_0'],
        "Rec_Normal": class_metrics['Rec_0'],
        "Prec_Fraud": class_metrics['Prec_1'],
        "Rec_Fraud": class_metrics['Rec_1'],
        "Macro_Recall": macro_recall,
        "F1_Score": metrics.weightedFMeasure(),
        "AUC": auc
    }
    return results