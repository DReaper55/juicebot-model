# CNN-Based Food and Drink Prediction Model

This project implements a Convolutional Neural Network (CNN) to predict drink composition based on food nutritional features. The dataset includes food features like calories, protein, fat, carbs, fiber, and sugar, and the labels are compositions of drinks.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Requirements](#requirements)
5. [Usage](#usage)
6. [Training the Model](#training-the-model)
7. [Evaluation](#evaluation)
8. [Results](#results)
9. [Contributing](#contributing)
10. [License](#license)

---

## Project Overview
This project aims to build a CNN model to predict drink compositions based on the nutritional content of paired foods. The model is built using TensorFlow/Keras, and includes regularization techniques like dropout and batch normalization to improve generalization.

---

## Dataset
The dataset is a JSON file structured as follows:
- **Features**:
    - `calories`, `protein`, `fat`, `carbs`, `fiber`, `sugar`
- **Labels**:
    - `composition` (4 values representing drink components of spicy, sweet, sour and bitter)

Example JSON structure:
```json
{
  "dataset": [
    {
      "drink": {
        "composition": [0.1, 0.3, 0.4, 0.2]
      },
      "food": [
        {
          "calories": 100,
          "protein": 5,
          "fat": 2,
          "carbs": 20,
          "fiber": 3,
          "sugar": 8
        }
      ]
    }
  ]
}
```

---

## Model Architecture
The CNN model consists of:
1. **Input Layer**: Accepts a 1D array of size `(6,)`.
2. **Hidden Layers**:
    - Two 1D convolutional layers with ReLU activation.
    - Dropout and Batch Normalization for regularization.
3. **Output Layer**: A dense layer with Softmax activation to predict 4 composition values.

---

## Requirements
Install the following dependencies before running the project:
- Python >= 3.7
- TensorFlow >= 2.8
- NumPy
- Matplotlib
- Scikit-learn

Install dependencies using pip:
```bash
pip install tensorflow numpy matplotlib scikit-learn
```

---

## Usage

### 1. Preprocessing the Dataset
Ensure your dataset is in JSON format. Use the preprocessing script to extract features and labels:
```python
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load and preprocess dataset
# Code from the project to preprocess features and labels.
```

### 2. Train the Model
Run the training script:
```bash
python train_model.py
```

### 3. Evaluate the Model
Evaluate the model on test data:
```bash
python evaluate_model.py
```

---

## Training the Model
To train the model:
1. Load the JSON dataset.
2. Preprocess it to create `X` (features) and `y` (labels).
3. Normalize the features using `StandardScaler`.
4. Train the CNN model:
    - Optimizer: Adam
    - Loss: Mean Absolute Error
    - Metrics: MAE
5. Monitor training and validation loss to detect overfitting or underfitting.

---

## Evaluation
Evaluate the model using test data:
- Metrics: Mean Absolute Error (MAE)
- Compare training and validation losses to assess generalization.

Example:
```bash
Test Loss: 0.234
Test MAE: 0.21
```

---

## Results
- The model achieved a **MAE of 0.21** on the test dataset.
- Visualization of training and validation losses over epochs:

![Training vs Validation Loss](assets/loss_plot.png)

---

## Contributing
Contributions are welcome! If you'd like to improve the model or add features:
1. Fork the repository.
2. Create a feature branch.
3. Submit a pull request.

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to customize this README further based on the specifics of your project!