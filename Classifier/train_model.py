import h5py
import numpy as np

from keras.src.models import Sequential
from keras.src.layers import Conv1D, Flatten, Dense, BatchNormalization, Dropout
from keras.src.optimizers import Adam
from keras.src.losses import MeanAbsoluteError
from keras.src.regularizers import L2

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

# Flatten the nested structure to create a flat DataFrame
flat_dataset = []
with h5py.File("dataset.h5", "r") as h5f:
    X = h5f["features"][:]
    y = h5f["labels"][:]

X = np.array(X)
y = np.array(y)

# print(X.shape)
# print(y.shape)

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_reshaped = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))

# Save the scalar mean and scale
# np.save('scaler_mean.npy', scaler.mean_)
# np.save('scaler_scale.npy', scaler.scale_)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.3, random_state=30)

# Select the first feature for plotting
# X_train_feature = X_train[:, 0]
# Y_train_feature = X_train[:, 2]

# Plot the feature against the target variable
# plt.figure(figsize=(10, 6))
# plt.scatter(X_train_feature, Y_train_feature, color='blue', alpha=0.5, label='Training Data')
# plt.xlabel("Feature 0 (e.g. Calories)")
# plt.ylabel("Label (Sour)")
# plt.title("Training Data: Feature vs. Composition")
# plt.legend()
# plt.show()

# Define the CNN model
model = Sequential()
model.add(Conv1D(32, kernel_size=2, activation='relu', kernel_regularizer=L2(0.001), input_shape=(X_train.shape[1], 1)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Conv1D(64, kernel_size=2, activation='relu', kernel_regularizer=L2(0.001)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv1D(32, kernel_size=2, activation='relu', kernel_regularizer=L2(0.001)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Flatten())
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(4, activation='linear'))  # Output layer with 4 units for each label value

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0005), loss=MeanAbsoluteError(), metrics=['mae'])

# Train the model
history = model.fit(X_train, y_train, epochs=200, batch_size=8, validation_data=(X_test, y_test))

# Evaluate the model on the test set
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}, Test MAE: {test_mae}')

# plt.plot(history.history['loss'], label='Train Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.title('Training and Validation Loss over Epochs')
# plt.show()

# Save the entire model (architecture and weights)
# model.save('trained_model2.keras')
# model.save('trained_model.h5')

# Make predictions for a new set of features
# new_features = np.array([[120.0, 8.0, 4.0, 12.0, 4.0, 2.0]])
# new_features = scaler.transform(new_features)
# new_features = new_features.reshape((1, new_features.shape[1], 1))
# predicted_composition = model.predict(new_features)

# print(f'Predicted ETA for the new features: {predicted_composition[0]}')
