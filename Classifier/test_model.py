import json
import os

import numpy as np

from keras.src.models import load_model
from sklearn.preprocessing import StandardScaler

current_dir = os.path.dirname(os.path.realpath(__file__))


def get_drinks():
    # Get the drink compositions from the drink dataset
    with open('drinks.json', 'r') as f:
        drink_data = json.load(f)
    return np.array([{'name': drink['name'], 'composition': drink['composition']} for drink in drink_data['drinks']])


def predict_eta(features):
    # Get files using absolute path
    model_path = os.path.join(current_dir, 'trained_model.h5')
    scalar_mean_path = os.path.join(current_dir, 'scaler_mean.npy')
    scalar_scale_path = os.path.join(current_dir, 'scaler_scale.npy')

    # Load the saved model
    loaded_model = load_model(model_path)

    # Load the mean and scale values from numpy files
    scaler_mean = np.load(scalar_mean_path)
    scaler_scale = np.load(scalar_scale_path)

    scaler = StandardScaler()
    scaler.mean_ = scaler_mean
    scaler.scale_ = scaler_scale

    # Use the loaded_model to make predictions
    new_features = np.array([features])

    # Transform the new features using the loaded scaler
    new_features = scaler.transform(new_features)
    new_features = new_features.reshape((1, new_features.shape[1], 1))

    # Make predictions with the loaded model
    predicted_eta = loaded_model.predict(new_features)

    return predicted_eta[0]


if __name__ == "__main__":
    predicted_eta_result = predict_eta([100.0, 6.0, 2.0, 14.0, 1.0, 0.0])
    print({"eta": predicted_eta_result})

    drinks = get_drinks()

    distances = [np.linalg.norm(np.array(predicted_eta_result) - np.array(drink['composition'])) for drink in drinks]
    closest_index = np.argmin(distances)

    print(drinks[closest_index]['name'])
    print(closest_index)
