"""
Inference service for the wind turbine power prediction model.
"""
from fastapi import FastAPI
from pre_process import preprocess_input
from schemas import BatchInput
from utility import load_model_from_s3, score_model


# Define the FastAPI app
app = FastAPI()

# Load the model (assume model.pkl is in the current directory)
model = load_model_from_s3("mlops-aws-windoutput")


# Define a root endpoint
@app.get("/")
def read_root():
    """
    Root endpoint for the FastAPI app.
    """
    return {"message": "Welcome to the Prediction API"}


@app.post("/predict")
def predict(batch_input: BatchInput):
    """
    Endpoint for making predictions on a batch of input data."""
    # Prepare a list for processed input values
    input_data = []

    for data in batch_input.inputs:
        # Append the values to the input_data list
        input_data.append(
            {
                "date_time": data.date_time,
                "wind_speed": data.wind_speed,
                "theoretical_power": data.theoretical_power,
                "wind_direction": data.wind_direction,
            }
        )

    # Wrap input_data in the expected format
    formatted_data = {"inputs": input_data}

    # Pass the formatted input data through a preprocessing function
    processed_data = preprocess_input(formatted_data)

    predictions = score_model(model, processed_data)

    # Return the predictions as a JSON response
    return {"predictions": predictions}


# Run the FastAPI app using Uvicorn (a production-ready ASGI server)
# if __name__ == "__main__":
#     import uvicorn

#     uvicorn.run(app, host="0.0.0.0", port=8000)
