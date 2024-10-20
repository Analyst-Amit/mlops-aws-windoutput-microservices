""" Define the input format for the model """
from typing import List

from pydantic import BaseModel


# Define the input format for each individual data row
class ModelInput(BaseModel):
    """Input data model for the wind turbine power prediction model."""

    date_time: str
    wind_speed: float
    theoretical_power: float
    wind_direction: float


# Define the input format for a batch of inputs
class BatchInput(BaseModel):
    """Input data model for a batch of wind turbine power prediction model inputs."""

    inputs: List[ModelInput]
