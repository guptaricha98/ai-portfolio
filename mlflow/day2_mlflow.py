import mlflow

mlflow.set_experiment("demo-exp")

with mlflow.start_run():
    mlflow.log_param("model", "all-MiniLM-L6-v2")
    mlflow.log_metric("recall_at_5", 0.72)

print("Run logged ✅ — start UI with: mlflow ui")
