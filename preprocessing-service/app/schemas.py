from typing import List

from pydantic import BaseModel


# Define the input format for each individual data row
class ModelInput(BaseModel):
    date_time: str
    wind_speed: float
    theoretical_power: float
    wind_direction: float


# Define the input format for a batch of inputs
class BatchInput(BaseModel):
    inputs: List[ModelInput]
