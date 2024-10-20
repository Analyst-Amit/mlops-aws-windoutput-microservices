from typing import Dict, List

import pandas as pd


def validate_columns(df: pd.DataFrame, required_columns: List[str]):
    """Validate if the required columns are present in the DataFrame."""
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")


def sanitize_input(data: List[Dict[str, float]]) -> List[Dict[str, float]]:
    """Prepare and clean the data and return it as a list of dictionaries.

    Args:
        data (List[Dict[str, float]]): The list of records to prepare.

    Returns:
        List[Dict[str, float]]: The processed data ready to be sent as JSON.
    """
    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(data)

    # Ensure the date_time column is parsed and added to the DataFrame
    df["date_time"] = pd.to_datetime(df["date_time"], format="%Y-%m-%dT%H:%M:%S")

    # Extract Month and Hour from the Date/Time
    df["month"] = df["date_time"].dt.month
    df["hour"] = df["date_time"].dt.hour

    # Remove the Date/Time column if it's no longer needed
    df.drop("date_time", axis=1, inplace=True)

    # Prepare the required columns
    required_columns = [
        "wind_speed",
        "theoretical_power",
        "wind_direction",
        "month",
        "hour",
    ]

    # Validate that all required columns are present
    validate_columns(df, required_columns)

    # Remove rows with months January or December
    df = df[~df["month"].isin([1, 12])]

    # Remove outliers based on wind speed
    Q1 = df["wind_speed"].quantile(0.25)
    Q3 = df["wind_speed"].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df["wind_speed"] < (Q1 - 1.5 * IQR)) | (df["wind_speed"] > (Q3 + 1.5 * IQR)))]

    # Convert the DataFrame to a list of dictionaries
    processed_data = df.to_dict(orient="records")
    print(processed_data)

    return processed_data


# This is the data format which should go inside pre-process function
# input_json = [
#     {
#         "date_time": "2024-10-18T12:00:00",
#         "wind_speed": 12.5,
#         "theoretical_power": 500,
#         "wind_direction": 90
#     },
#     {
#         "date_time": "2024-10-18T13:00:00",
#         "wind_speed": 14.3,
#         "theoretical_power": 550,
#         "wind_direction": 100
#     }
# ]

# sanitize_input(input_json)
