# PJME Hourly Energy Consumption Forecasting

This project focuses on forecasting hourly energy consumption for the PJM East region using historical data. It leverages machine learning techniques, specifically a **Random Forest Regressor**, to predict future energy demand based on time-based features.

## Project Structure

The project directory contains the following files:

- **`timeseries (3).ipynb`**: The main Jupyter Notebook containing the entire data science workflow:
    - Data loading and exploratory data analysis (EDA).
    - Data cleaning and preprocessing.
    - Feature engineering (extracting temporal features).
    - Model training and hyperparameter tuning using `RandomizedSearchCV`.
    - Evaluation and visualization of results.
- **`PJME_hourly.csv`**: The raw dataset containing hourly energy consumption (in Megawatts) for the PJM East region.
    - Columns: `Datetime`, `PJME_MW`.
- **`train_clean.csv`**: Preprocessed training dataset (saved intermediate file).
- **`test_clean.csv`**: Preprocessed testing dataset (saved intermediate file).
- **`energy_forecast_model.pkl`**: The trained and tuned Random Forest model saved for future inference.
- **`README.md`**: This documentation file.

## Data Analysis & Modeling Workflow

The notebook `timeseries (1).ipynb` executes the following steps:

1.  **Data Preparation**:
    - Loads the raw data `PJME_hourly.csv`.
    - Sets the `Datetime` column as the index.
    - Cleans the data (handling duplicates, missing values).

2.  **Feature Engineering**:
    - Extracts meaningful temporal features from the datetime index to capture seasonal and daily patterns:
        - `hour`: Captures daily cycles (peak hours vs. off-peak).
        - `dayofweek`: Captures weekly patterns (weekdays vs. weekends).
        - `quarter`, `month`: Captures seasonal variations.
        - `year`: Captures long-term trends.
        - `dayofyear`: Captures annual cycles.

3.  **Train/Test Split**:
    - The data is split chronologically to respect the time series nature of the problem.
    - **Training Set**: Data prior to `01-01-2015`.
    - **Test Set**: Data from `01-01-2015` onwards.

4.  **Model Training**:
    - **Algorithm**: Random Forest Regressor (`sklearn.ensemble.RandomForestRegressor`).
    - **Hyperparameter Tuning**: Uses `RandomizedSearchCV` to find the best parameters:
        - `n_estimators`: Number of trees in the forest.
        - `max_depth`: Maximum depth of the tree.
        - `min_samples_leaf`: Minimum number of samples required to be at a leaf node.
    - **Cross-Validation**: 5-fold cross-validation (`cv=3` was used in the final run based on code).

5.  **Evaluation**:
    - The model's performance is evaluated on the unseen test set using:
        - **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual values.
        - **RMSE (Root Mean Squared Error)**: Square root of the average squared differences, penalizing larger errors more.
    - **Feature Importance**: Visualizes which features (e.g., `hour`, `month`) have the most impact on predictions.

6.  **Saving the Model**:
    - The best-performing model is saved as `energy_forecast_model.pkl` using `joblib`.

## Installation & Usage

### Prerequisites

Ensure you have Python installed along with the following libraries:

```bash
pip install pandas numpy matplotlib scikit-learn joblib notebook
```

### Running the Project

1.  Clone this repository or download the project files.
2.  Navigate to the project directory:
    ```bash
    cd <path-to-folder>
    ```
3.  Launch Jupyter Notebook:
    ```bash
    jupyter notebook "timeseries (1).ipynb"
    ```
4.  Run all cells in the notebook to reproduce the analysis, train the model, and view the results.

### Using the Saved Model

You can load the saved model in any Python script to make predictions:

```python
import joblib
import pandas as pd

# Load the model
model = joblib.load("energy_forecast_model.pkl")

# Prepare input data (ensure it has the same features: hour, dayofweek, etc.)
sample_data = pd.DataFrame({
    'hour': [14],
    'dayofweek': [2],
    'quarter': [3],
    'month': [8],
    'year': [2024],
    'dayofyear': [225]
})

# Predict
prediction = model.predict(sample_data)
print(f"Predicted Energy Consumption: {prediction[0]} MW")
```

## Results

Key insights from the analysis:
- **Feature Importance**: The model identifies `hour`, `dayofyear`, and `month` as significant predictors, reflecting strong daily and seasonal energy usage patterns.
- **Model Performance**: (Refer to the notebook output for final MAE and RMSE values).

