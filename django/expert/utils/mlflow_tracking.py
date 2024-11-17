import mlflow
import mlflow.pytorch
from datetime import datetime

class MLFlowTracker:
    def __init__(self, experiment_name="hate_detection"):
        mlflow.set_tracking_uri("file:./expert/mlflow/experiments")
        self.experiment = mlflow.set_experiment(experiment_name)
        
    def start_run(self, run_name=None):
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return mlflow.start_run(experiment_id=self.experiment.experiment_id, run_name=run_name)
    
    def log_params(self, params):
        mlflow.log_params(params)
    
    def log_metrics(self, metrics):
        mlflow.log_metrics(metrics)
    
    def log_model(self, model, model_name):
        mlflow.pytorch.log_model(model, model_name)
    
    def end_run(self):
        mlflow.end_run()