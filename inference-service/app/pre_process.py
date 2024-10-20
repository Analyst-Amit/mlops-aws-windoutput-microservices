""" This module contains the function to send data to the preprocessing API. """
import requests


def preprocess_input(data):
    """
    Sends data to the preprocessing API and returns the processed data.

    Args:
        data (dict): The input data to be preprocessed.

    Returns:
        dict: The processed data returned by the API.
    """

    api_url = "http://preprocessing-service:8001/preprocess"
    print(f"Sending preprocessing data to API: {api_url}")

    try:
        # Set a timeout for the request (e.g., 5 seconds)
        response = requests.post(api_url, json=data, timeout=15)
        response.raise_for_status()

        processed_data = response.json().get("processed_data")
        if processed_data is None:
            raise ValueError("Processed data not found in the response.")
        return processed_data

    except requests.RequestException as e:
        print(f"An error occurred while sending data to the API: {e}")
        raise
    except ValueError as e:
        print(f"An error occurred while processing the response: {e}")
        raise
