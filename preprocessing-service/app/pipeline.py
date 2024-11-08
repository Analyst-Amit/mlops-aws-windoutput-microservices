from typing import Dict, List

import pandas as pd
from sklearn.model_selection import train_test_split


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
    print(df.head())
    # Ensure the date_time column is parsed and added to the DataFrame
    df["date_time"] = pd.to_datetime(df["date_time"], format="%d %m %Y %H:%M")

    # Extract Month and Hour from the Date/Time
    df["month"] = df["date_time"].dt.month.astype(float)
    df["hour"] = df["date_time"].dt.hour.astype(float)
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

    return processed_data


def split_data(data, test_size: float = 0.2):
    df = pd.DataFrame(sanitize_input(data))

    df = remove_invalid_power_rows(df)
    trainDF, testDF = train_test_split(df, test_size=test_size, random_state=1234)
    X_train = trainDF.drop(columns=["active_power"]).values.tolist()
    y_train = trainDF["active_power"].values.tolist()
    X_test = testDF.drop(columns=["active_power"]).values.tolist()
    y_test = testDF["active_power"].values.tolist()

    return {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test}


def remove_invalid_power_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where LV ActivePower is 0 but Theoretical_Power_Curve is not 0.

    Args:
        df (pd.DataFrame): The DataFrame to clean.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    df = df[~((df["active_power"] == 0) & (df["theoretical_power"] != 0))]
    return df


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
