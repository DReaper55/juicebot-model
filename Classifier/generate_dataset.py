import json

import h5py
import numpy as np

with open("data1.json", "r") as json_file:
    data = json.load(json_file)

# Prepare datasets for storing
features_data = []
labels_data = []

for entry in data['dataset']:
    drink = entry['drink']
    for food in entry['food']:
        # Flatten the drink and food features for each food entry
        features = [
            food["calories"],
            food["protein"],
            food["fat"],
            food["carbs"],
            food["fiber"],
            food["sugar"]
        ]
        labels = [
            drink["composition"][0],
            drink["composition"][1],
            drink["composition"][2],
            drink["composition"][3],
        ]
        features_data.append(features)
        labels_data.append(labels)

# Convert lists to numpy arrays
features_data = np.array(features_data, dtype=np.float32)
labels_data = np.array(labels_data, dtype=np.float32)

with h5py.File("dataset.h5", "w") as h5f:
    h5f.create_dataset("features", data=features_data)
    h5f.create_dataset("labels", data=labels_data)


# Load the .npy files
mean = np.load('scaler_mean.npy')
scale = np.load('scaler_scale.npy')

# Convert to lists and save as JSON
mean_list = mean.tolist()
scale_list = scale.tolist()

with open('mean.json', 'w') as mean_file:
    json.dump(mean_list, mean_file)

with open('scale.json', 'w') as scale_file:
    json.dump(scale_list, scale_file)


# print(features_data.shape)
# print(labels_data.shape)
