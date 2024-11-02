FROM python:3.9-slim

# Install MLflow
RUN pip install mlflow==2.17.2
RUN pip install boto3 psycopg2-binary

# Set environment variables for AWS and MLflow
ENV MLFLOW_S3_ENDPOINT_URL=http://minio:9000
ENV AWS_ACCESS_KEY_ID=minio_user
ENV AWS_SECRET_ACCESS_KEY=minio_password

# Expose the port that MLflow uses
EXPOSE 5000

# Set the command to run the MLflow server with the desired parameters
CMD ["mlflow", "server", "--backend-store-uri", "postgresql://user:password@postgres:5432/mlflowdb", "--artifacts-destination", "s3://mlops-aws-windoutput-staging", "--host", "0.0.0.0", "--port", "5000"]
