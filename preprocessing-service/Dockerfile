# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY ./app /app/
COPY requirements.txt /app/requirements.txt

# Install any necessary dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Expose the port that the app will run on
EXPOSE 8001

# Run the FastAPI app using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
