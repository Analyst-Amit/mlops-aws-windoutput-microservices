"""Utility functions for the inference service."""
import tempfile
from typing import Any, Dict, List

import boto3
import joblib
from botocore.exceptions import ClientError


# Scoring function to make predictions for multiple rows
def score_model(model: Any, inputs: List[Dict[str, float]]) -> List[float]:
    """Score multiple sets of inputs using the loaded model."""
    # Extract the feature values from the input dictionaries
    formatted_inputs = [
        [
            record["wind_speed"],
            record["theoretical_power"],
            record["wind_direction"],
            record["month"],
            record["hour"],
        ]
        for record in inputs
    ]

    # Make predictions using the model
    return model.predict(formatted_inputs).tolist()


def load_model_from_s3(bucket_name: str) -> Any:
    """Load the model from an S3 bucket."""
    s3_client = boto3.client("s3")
    key = "Artifacts/model.bin"
    """Load the model from S3."""
    model = None
    try:
        with tempfile.TemporaryFile() as fp:
            s3_client.download_fileobj(Fileobj=fp, Bucket=bucket_name, Key=key)
            fp.seek(0)
            model = joblib.load(fp)
            print(f"Model loaded from s3://{bucket_name}/{key}")
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        print(error_code)
        print(f"Failed to load model from S3" f": Model not found at s3://{bucket_name}/Artifacts")
        return "404"

    return model
