""" This module defines the FastAPI application for the preprocessing service. """
from fastapi import FastAPI, HTTPException
from pipeline import sanitize_input, split_data
from schemas import BatchInput, BatchTrainInput


# Define the FastAPI app
app = FastAPI()


# Define a root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Prediction API"}


# Define an endpoint for real-time preprocessing
@app.post("/preprocess")
def preprocess(data: BatchInput):
    try:
        # Extract list of dictionaries from the BatchInput model
        input_data = [record.dict() for record in data.inputs]

        # Preprocess the input data using the sanitize_input function
        processed_data = sanitize_input(input_data)

        # Return the processed data as a JSON response
        return {"processed_data": processed_data}
    except Exception as e:
        # Log the error and raise an HTTP exception
        print(f"Error occurred: {e}")
        raise HTTPException(status_code=500, detail="Error while preprocessing the data")


# Define an endpoint for real-time preprocessing
@app.post("/preprocess_train")
def preprocess_train(data: BatchTrainInput):
    try:
        # Extract list of dictionaries from the BatchInput model
        input_data = [record.dict() for record in data.inputs]

        # Preprocess the input data using the sanitize_input function
        processed_data = split_data(input_data)

        # Return the processed data as a JSON response
        return {"processed_data": processed_data}
    except Exception as e:
        # Log the error and raise an HTTP exception
        print(f"Error occurred: {e}")
        raise HTTPException(status_code=500, detail="Error while preprocessing the data")


# if __name__ == "__main__":
#     import uvicorn

#     uvicorn.run(app, host="0.0.0.0", port=8001)


# This format needs to be passed.
# {
#     "inputs": [
#         {
#             "date_time": "2024-10-18T12:00:00",
#             "wind_speed": 12.5,
#             "theoretical_power": 500.0,
#             "wind_direction": 90.0
#         },
#         {
#             "date_time": "2024-10-18T13:00:00",
#             "wind_speed": 14.3,
#             "theoretical_power": 550.0,
#             "wind_direction": 100.0
#         }
#     ]
# }
