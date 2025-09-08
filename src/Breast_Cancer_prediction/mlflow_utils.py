import mlflow

# Minimal MLflow helpers mirroring your notebook usage

def set_experiment(name: str) -> None:
    mlflow.set_experiment(name)


def set_tracking_uri(uri: str) -> None:
    mlflow.set_tracking_uri(uri)


def log_pandas_dataset(df, name: str, targets: str | None = None):
    import mlflow.data
    dataset = mlflow.data.from_pandas(df, name=name, targets=targets)
    mlflow.log_input(dataset)
    return dataset


def start_run(run_name: str):
    return mlflow.start_run(run_name=run_name)


def log_params(params: dict) -> None:
    if params:
        mlflow.log_params(params)


def log_metrics(metrics: dict) -> None:
    if metrics:
        mlflow.log_metrics(metrics)


def log_sklearn_model(sk_model, name: str, input_example=None, registered_model_name: str | None = None) -> None:
    import mlflow.sklearn
    mlflow.sklearn.log_model(sk_model=sk_model, artifact_path=name, input_example=input_example, registered_model_name=registered_model_name)


def log_xgboost_model(model, name: str, input_example=None, registered_model_name: str | None = None) -> None:
    import mlflow.xgboost
    mlflow.xgboost.log_model(model, artifact_path=name, input_example=input_example, registered_model_name=registered_model_name)


def log_artifact(local_path: str) -> None:
    mlflow.log_artifact(local_path)
