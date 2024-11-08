"""
Inference service for the wind turbine power prediction model.
"""

from fastapi import FastAPI, HTTPException
from pre_process import preprocess_input
from schemas import BatchInput
from utility import load_model, score_model


# Initialize FastAPI app
app = FastAPI()

# Load the model when the app starts
model = load_model()
print("Model Load successful")


# Root endpoint
@app.get("/")
def read_root():
    """Root endpoint for the FastAPI app."""
    return {"message": "Welcome to the Prediction API"}


@app.post("/predict")
def predict(batch_input: BatchInput):
    """
    Endpoint for making predictions on a batch of input data.
    """

    if model is None:
        raise HTTPException(status_code=503, detail="Model not available. Please try again later.")

    # Format input data for preprocessing
    input_data = [
        {
            "date_time": data.date_time,
            "wind_speed": data.wind_speed,
            "theoretical_power": data.theoretical_power,
            "wind_direction": data.wind_direction,
        }
        for data in batch_input.inputs
    ]

    # Preprocess and score the data
    try:
        processed_data = preprocess_input({"inputs": input_data})
        print(processed_data)
        predictions = score_model(model, processed_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return {"predictions": predictions}


# # Uncomment below to run with Uvicorn directly if needed
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
