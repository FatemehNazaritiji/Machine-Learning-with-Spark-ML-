# Enhanced Real Estate Modeling with Apache Spark MLlib

This project focuses on using Apache Spark's MLlib to perform machine learning on a real estate dataset. It leverages the capabilities of Spark's DataFrame and RandomForestRegressor for efficient data processing and model training.

## Project Overview

The goal of this project is to build and optimize a machine learning model to predict real estate prices per unit area. The project involves the following steps:

1. **Data Preprocessing**: Load and preprocess the dataset by engineering new features.
2. **Pipeline Construction**: Build a machine learning pipeline using Spark's `Pipeline`, `VectorAssembler`, and `StandardScaler`.
3. **Model Training and Tuning**: Train a `RandomForestRegressor` model using cross-validation to find the best parameters.
4. **Model Evaluation**: Evaluate the model's performance using regression metrics such as RMSE, MAE, MSE, and R-squared.

## Dataset

The dataset used in this project is a real estate dataset provided in the `realestate.csv` file. Key features include:

- `HouseAge`: Age of the house in years.
- `DistanceToMRT`: Distance to the nearest MRT station in meters.
- `NumberConvenienceStores`: Number of convenience stores near the house.
- `PriceOfUnitArea`: Price of the real estate per unit area.

## Features Engineered

- `LogDistanceToMRT`: Log transformation of the `DistanceToMRT` feature.
- `HouseAgeSquared`: Square of the `HouseAge` feature.
- `Interaction`: Interaction term between `LogDistanceToMRT` and `NumberConvenienceStores`.

## Installation and Setup

1. **Clone the repository**:
    ```bash
    git clone https://github.com/FatemehNazaritiji/Machine-Learning-with-Spark-ML-.git
    cd Machine-Learning-with-Spark-ML-
    ```

2. **Prerequisites**:
    - Apache Spark installed on your machine.
    - Python 3.x with pip installed.

3. **Install dependencies**:
    This project assumes Spark is installed and running. You can set up a Python environment and install additional dependencies if necessary.

4. **Run the script**:
    Execute the main script using Spark's `spark-submit`:
    ```bash
    spark-submit real_estate.py
    ```

## Key Learnings

Through this project, I have enhanced my skills in:

- Data preprocessing and feature engineering with Spark.
- Building and tuning machine learning models using Spark's MLlib.
- Using Spark's DataFrame API for efficient data processing.
- Implementing cross-validation and hyperparameter tuning for model optimization.

## Results

The project outputs the model's performance metrics, including RMSE, MAE, MSE, and R-squared, to assess the accuracy and reliability of the predictions.
- **Root Mean Squared Error (RMSE):** 6.4973
- **Mean Absolute Error (MAE):** 4.0632
- **Mean Squared Error (MSE):** 42.2147
- **R-squared (R2):** 0.7990

## Contributing

If you'd like to contribute to this project, please fork the repository and submit a pull request.

