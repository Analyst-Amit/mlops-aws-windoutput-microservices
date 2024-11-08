"""Utility functions for the inference service."""
# import tempfile
from typing import Any, Dict, List, Union

# import boto3
# import joblib
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient


# from botocore.exceptions import ClientError


def score_model(model: Any, inputs: List[Dict[str, Union[float, int]]]) -> List[float]:
    """Score multiple sets of inputs using the loaded model.

    Args:
        model (Any): The machine learning model used for prediction.
        inputs (List[Dict[str, Union[float, int]]]): A list of input dictionaries,
            each with keys: "wind_speed", "theoretical_power", "wind_direction", "month", "hour".

    Returns:
        List[float]: The predicted values for each input.
    """
    # Creating a DataFrame with columns the model expects
    input_df = pd.DataFrame(
        [
            {
                "wind_speed": record["wind_speed"],
                "theoretical_power": record["theoretical_power"],
                "wind_direction": record["wind_direction"],
                "month": record["month"],
                "hour": record["hour"],
            }
            for record in inputs
        ]
    )
    return model.predict(input_df).tolist()


def load_model_by_alias(model_name, alias):
    """
    Load a model from MLflow registry using its alias.
    """
    model_uri = f"models:/{model_name}@{alias}"
    return mlflow.pyfunc.load_model(model_uri)


def load_model():
    """
    Loads the model from the MLflow model registry.
    Returns the model if loaded successfully; otherwise, returns None.
    """
    # Configuration for MLflow tracking URI and model details
    MLFLOW_TRACKING_URI = "http://mlflow-server:5000"
    MODEL_NAME = "sk-learn-extra-trees-regression-model-wind-output"
    MODEL_ALIAS = "champion"

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    try:
        # Check if model is available in MLflow registry
        client.search_model_versions(f"name='{MODEL_NAME}'")
        model = load_model_by_alias(MODEL_NAME, MODEL_ALIAS)
        print("Connected to MLflow and model loaded successfully.")
        return model
    except Exception as e:
        print("Failed to connect to MLflow server:", e)
        return None  # Model could not be loaded


# # Scoring function to make predictions for multiple rows
# def score_model(model: Any, inputs: List[Dict[str, float]]) -> List[float]:
#     """Score multiple sets of inputs using the loaded model."""
#     # Extract the feature values from the input dictionaries
#     formatted_inputs = [
#         [
#             record["wind_speed"],
#             record["theoretical_power"],
#             record["wind_direction"],
#             record["month"],
#             record["hour"],
#         ]
#         for record in inputs
#     ]

#     # Make predictions using the model
#     return model.predict(formatted_inputs).tolist()

# def score_model_new(
#     model: Any,
#     wind_speed: float,
#     theoretical_power: float,
#     wind_direction: float,
#     month: int,
#     hour: int,
# ) -> float:
#     """Score a single set of inputs using the loaded model.

#     Args:
#         model (Any): The machine learning model used for prediction.
#         wind_speed (float): The wind speed value.
#         theoretical_power (float): The theoretical power curve value.
#         wind_direction (float): The wind direction value.
#         month (int): The month of the observation.
#         hour (int): The hour of the observation.

#     Returns:
#         float: The predicted value from the model.
#     """
#     # return model.predict([[wind_speed, theoretical_power, wind_direction, month, hour]])[0]
#     # Creating a DataFrame with named columns for the model input
#     input_df = pd.DataFrame(
#         [
#             {
#                 "Wind Speed (m/s)": wind_speed,
#                 "Theoretical_Power_Curve (KWh)": theoretical_power,
#                 "Wind Direction (°)": wind_direction,
#                 "Month": month,
#                 "Hour": hour,
#             }
#         ]
#     )
#     return model.predict(input_df)[0]
# def load_model_from_s3(bucket_name: str) -> Any:
#     """Load the model from an S3 bucket."""
#     s3_client = boto3.client("s3")
#     key = "Artifacts/model.bin"
#     """Load the model from S3."""
#     model = None
#     try:
#         with tempfile.TemporaryFile() as fp:
#             s3_client.download_fileobj(Fileobj=fp, Bucket=bucket_name, Key=key)
#             fp.seek(0)
#             model = joblib.load(fp)
#             print(f"Model loaded from s3://{bucket_name}/{key}")
#     except ClientError as e:
#         error_code = e.response["Error"]["Code"]
#         print(error_code)
#         print(f"Failed to load model from S3 Model not found at s3://{bucket_name}/Artifacts")
#         return "404"

#     return model


# def load_model_by_alias(model_name, alias):
#     """
#     Load a model from MLflow registry using its alias.
#     """
#     model_uri = f"models:/{model_name}@{alias}"
#     return mlflow.pyfunc.load_model(model_uri)
