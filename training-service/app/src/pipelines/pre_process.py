""" This module contains the function to send data to the preprocessing API. """
import pandas as pd
import requests


def preprocess_input(data):
    """
    Sends data to the preprocessing API and returns the processed data.

    Args:
        data (dict): The input data to be preprocessed.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        The training and testing features and targets as DataFrames.
    """

    api_url = "http://preprocessing-service:8001/preprocess_train"
    # api_url = "http://localhost:8001/preprocess_train"
    print(f"Sending preprocessing data to API: {api_url}")
    # print(data[:5])
    try:
        # Structure the input data as required by the API
        # payload = {"inputs": data}
        # Set a timeout for the request (e.g., 15 seconds)
        response = requests.post(api_url, json=data, timeout=15)
        # Extract processed data if the request was successful
        processed_data = response.json().get("processed_data")
        print(processed_data["X_train"][0:5])
        if not processed_data:
            raise ValueError("Processed data not found in response")

        # Define column names for the feature sets
        feature_columns = ["wind_speed", "theoretical_power", "wind_direction", "month", "hour"]

        # Convert lists back to DataFrames
        X_train = pd.DataFrame(processed_data["X_train"], columns=feature_columns)
        y_train = pd.DataFrame(
            processed_data["y_train"], columns=["active_power"]
        ).values.ravel()  # Convert to 1D array
        X_test = pd.DataFrame(processed_data["X_test"], columns=feature_columns)
        y_test = pd.DataFrame(
            processed_data["y_test"], columns=["active_power"]
        ).values.ravel()  # Convert to 1D array

        # # Convert 'month' and 'hour' columns to integer type using astype
        # X_train['month'] = X_train['month'].astype(int)
        # X_train['hour'] = X_train['hour'].astype(int)

        # X_test['month'] = X_test['month'].astype(int)
        # X_test['hour'] = X_test['hour'].astype(int)

        return X_train, y_train, X_test, y_test

    except requests.RequestException as e:
        print(f"An error occurred while sending data to the API: {e}")
        raise
    except ValueError as e:
        print(f"An error occurred while processing the response: {e}")
        raise
