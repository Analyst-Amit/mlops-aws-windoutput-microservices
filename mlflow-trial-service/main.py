# from pprint import pprint

# import mlflow
# from mlflow.tracking import MlflowClient


# def format_registered_models(models):
#     formatted_models = []
#     for model in models:
#         formatted_model = {
#             "name": model.name,
#             "description": model.description,
#             "creation_timestamp": model.creation_timestamp,
#             "last_updated_timestamp": model.last_updated_timestamp,
#             "aliases": model.aliases,
#             "latest_versions": [
#                 {
#                     "version": version.version,
#                     "current_stage": version.current_stage,
#                     "run_id": version.run_id,
#                     "source": version.source,
#                     "status": version.status,
#                     "tags": version.tags,
#                 }
#                 for version in model.latest_versions
#             ],
#         }
#         formatted_models.append(formatted_model)
#     return formatted_models

# # Set the tracking URI (replace with your actual URI)
# mlflow.set_tracking_uri("http://localhost:5000")  # or the appropriate URI

# # Confirm the tracking URI
# print("Tracking URI:", mlflow.get_tracking_uri())

# client = MlflowClient()


# models = client.list_registered_models()
# pprint(format_registered_models(models))


# import mlflow
# from mlflow.tracking import MlflowClient


# def load_model_from_registry(model_name, alias):
#     """
#     Loads a registered MLflow model by its alias and performs predictions.

#     Parameters:
#         model_name (str): The name of the registered model.
#         alias (str): The alias of the model version to load (e.g., 'Champion').

#     Returns:
#         array-like: Predictions from the model.
#     """
#     # Initialize the MLflow client
#     client = MlflowClient()

#     # Get the model version using the alias
#     model_version = client.get_model_version_by_alias(model_name, alias)

#     # Get the model URI for loading
#     model_uri = f"models:/{model_name}@{alias}"

#     # Load the model
#     model = mlflow.pyfunc.load_model(model_uri)

#     return model

# model_name = "sk-learn-extra-trees-regression-model-wind-output"
# challenger_model = load_model_from_registry(model_name, alias="challenger")
# # Prepare your data
# import pandas as pd


# data = pd.DataFrame({
#     "Wind Speed (m/s)": [8.218296051, 4.995032787, 2.212670088, 2.190548897, 4.157712936],
#     "Theoretical_Power_Curve (KWh)": [1657.373187, 334.7802577, 0, 0, 152.8500671],
#     "Wind Direction (°)": [78.12586975, 17.07011986, 127.6598969, 329.9155884, 81.20819855],
#     "Month": [8, 5, 4, 7, 4],
#     "Hour": [21, 11, 19, 11, 10]
# })
# # Perform prediction
# challenger_predictions = challenger_model.predict(data)


# actual = [1339.537964, 277.7774048, 0, 0, 87.54579926]


# import numpy as np
# from sklearn.metrics import mean_squared_error


# def calculate_rmse(predictions, true_values):
#     """
#     Calculates the Root Mean Squared Error (RMSE) between predictions and true values.

#     Parameters:
#         predictions (array-like): The predicted values from the model.
#         true_values (array-like): The actual target values.

#     Returns:
#         float: The RMSE value.
#     """
#     # Calculate RMSE
#     rmse = np.sqrt(mean_squared_error(true_values, predictions))
#     return rmse

# challenger_rmse = calculate_rmse(challenger_predictions, actual)
# print("RMSE:", rmse)

# champion_model = load_model_from_registry(model_name, alias="champion")
# # Perform prediction
# champion_predictions = champion_model.predict(data)
# champion_rmse = calculate_rmse(champion_predictions, actual)

#
# if challenger_rmse < champion_rmse:
#     print("Challenger model is better. Updating Champion model...")

#     # Get the current versions for 'champion' and 'challenger' aliases
#     challenger_version = client.get_model_version_by_alias(model_name, "challenger").version
#     champion_version = client.get_model_version_by_alias(model_name, "champion").version

#     # Set the 'challenger' model as the new 'champion'
#     client.set_registered_model_alias(model_name, "champion", challenger_version)

#     # Remove the 'challenger' alias from the current model version
#     client.delete_registered_model_alias(model_name, "challenger")

#     # Archive the previous 'champion' model
#     client.set_registered_model_alias(model_name, "archived", champion_version)

#     print("Champion model updated, and previous Champion model moved to Archive.")
# else:
#     print("Current Champion model is better; no update performed.")
#     # Set the challenger version as archived
#     # Get the current versions for 'challenger' aliases
#     challenger_version = client.get_model_version_by_alias(model_name, "challenger").version
#     client.set_registered_model_alias(model_name, "archived", challenger_version)

# model_name = "sk-learn-extra-trees-regression-model-wind-output"
# #To set, update, and delete aliases using the MLflow Client API, see the examples below:


# # reassign the "Champion" alias to version 2
# client.set_registered_model_alias(model_name, "champion", "2")

# # create "Archived" alias for version 3
# client.set_registered_model_alias(model_name, "challenger", "3")

# # get a model version by alias
# champion_version = client.get_model_version_by_alias(model_name, "champion")

# # delete the alias
# client.delete_registered_model_alias(model_name, "candidate")

# #Load model version by alias for inference workloads
# import mlflow.pyfunc


# model_version_uri = "models:/sk-learn-extra-trees-regression-model-wind-output@champion"
# champion_version = mlflow.pyfunc.load_model(model_version_uri)
# # champion_version.predict(test_x)


# #Rename a registered model
# client.rename_registered_model("<full-model-name>", "<new-model-name>")

# export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
# Replace with remote host name or IP address in an actual environment

# import mlflow
# from mlflow.tracking import MlflowClient


# client = MlflowClient()

# model_name = "sk-learn-extra-trees-regression-model-wind-output"
# model_versions = client.search_model_versions(f"name='{model_name}'")

# client.delete_model_version(name=model_name, version="1")


# import mlflow
# from sklearn.datasets import load_diabetes
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split


# mlflow.autolog()

# db = load_diabetes()
# X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

# # Create and train models.
# rf = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
# rf.fit(X_train, y_train)

# # Use the model to make predictions on the test dataset.
# predictions = rf.predict(X_test)
