# Real-Time Air Quality Prediction using Machine Learning

A machine learning project that analyzes real-time air quality data from Indian monitoring stations and predicts average pollutant concentration using environmental, geographic, and pollutant-level features. The project includes data cleaning, exploratory data analysis, feature selection, model benchmarking, and hyperparameter tuning.

![Model Performance Comparison](assets/model-performance-comparison.png)

## Project Overview

Air pollution is a major public health and environmental challenge, especially in rapidly urbanizing regions. This project uses real-time air quality data to understand pollution patterns across Indian cities and build regression models that estimate `pollutant_avg`, the average pollutant concentration reported by monitoring stations.

The workflow is designed as an end-to-end machine learning pipeline: raw data is inspected, missing values are handled, numerical features are scaled, categorical fields are encoded, exploratory visualizations are generated, and multiple regression models are compared using standard evaluation metrics.

## Objectives

- Analyze real-time air quality records across states, cities, and monitoring stations.
- Clean missing values and duplicate records to improve data quality.
- Explore pollutant distributions, outliers, and relationships between numerical features.
- Select predictive features using correlation analysis, SelectKBest, and Random Forest importance.
- Train and compare multiple regression models for pollutant prediction.
- Tune Random Forest hyperparameters using GridSearchCV.

## Dataset

The notebook uses `Real_time_air_quality.csv`, containing air quality observations from monitoring stations in India.

| Attribute | Description |
| --- | --- |
| Rows | 943 |
| Columns | 11 |
| Country | India |
| Target variable | `pollutant_avg` |
| Key numeric features | `latitude`, `longitude`, `pollutant_min`, `pollutant_max` |
| Key categorical features | `state`, `city`, `station`, `pollutant_id` |

Main columns:

- `country`, `state`, `city`, `station`
- `last_update`
- `latitude`, `longitude`
- `pollutant_id`
- `pollutant_min`, `pollutant_max`, `pollutant_avg`

## Methodology

### 1. Data Preprocessing

- Loaded the dataset using Pandas.
- Checked shape, data types, descriptive statistics, and missing values.
- Filled missing values in `pollutant_min`, `pollutant_max`, and `pollutant_avg` using median imputation.
- Removed duplicate rows.
- Converted `last_update` into datetime format.
- Applied one-hot encoding to categorical location fields.
- Scaled numerical columns using MinMaxScaler.

### 2. Exploratory Data Analysis

The project visualizes numerical feature distributions, pollutant relationships, correlations, and outliers.

![Numerical Feature Distributions](assets/numerical-feature-distributions.png)

The correlation heatmap shows that `pollutant_max` and `pollutant_min` are the strongest predictors of `pollutant_avg`.

![Correlation Heatmap](assets/correlation-heatmap.png)

A scatter plot between average and maximum pollutant values confirms a strong positive relationship.

![Pollutant Average vs Maximum](assets/pollutant-avg-vs-max.png)

### 3. Feature Selection

Three feature-selection approaches were explored:

- Correlation-based selection
- SelectKBest with `f_regression`
- Random Forest feature importance

Top features identified:

| Feature | Insight |
| --- | --- |
| `pollutant_max` | Strongest predictor of average pollutant level |
| `pollutant_min` | Second strongest predictor |
| `latitude`, `longitude` | Lower but useful geographic signal |

Random Forest feature importance ranked `pollutant_max` highest, followed by `pollutant_min`.

### 4. Model Training

The dataset was split into 80% training and 20% testing data with `random_state=42`. The following regression models were trained:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- K-Nearest Neighbors Regressor

## Results

| Model | MAE | MSE | RMSE | R2 Score |
| --- | ---: | ---: | ---: | ---: |
| Linear Regression | 0.020155 | 0.001053 | 0.032455 | 0.967068 |
| Random Forest | 0.021964 | 0.001757 | 0.041919 | 0.945063 |
| Gradient Boosting | 0.023389 | 0.001775 | 0.042126 | 0.944518 |
| KNN Regressor | 0.026766 | 0.002483 | 0.049829 | 0.922374 |
| Decision Tree | 0.028525 | 0.002679 | 0.051762 | 0.916232 |

Linear Regression achieved the best overall performance, with the highest R2 score and lowest RMSE on the test set.

![Linear Regression Actual vs Predicted](assets/linear-regression-actual-vs-predicted.png)

## Hyperparameter Tuning

Random Forest was tuned using GridSearchCV with 3-fold cross-validation.

Best parameters:

```python
{
    'max_depth': 10,
    'min_samples_leaf': 2,
    'min_samples_split': 2,
    'n_estimators': 50
}
```

## Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## How to Run

1. Clone the repository.

```bash
git clone https://github.com/<your-username>/<your-repository>.git
cd <your-repository>
```

2. Install dependencies.

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

3. Place the dataset in the project directory.

```text
Real_time_air_quality.csv
```

4. Open the notebook.

```bash
jupyter notebook "ML Project.ipynb"
```

5. Run all cells to reproduce preprocessing, EDA, model training, and evaluation.


```

## Key Takeaways

- `pollutant_max` and `pollutant_min` are the most influential predictors of average pollutant concentration.
- Linear Regression performed best, indicating a strong linear relationship between pollutant minimum, maximum, and average values.
- Ensemble models such as Random Forest and Gradient Boosting also performed strongly and can be useful for more complex future datasets.
- The project demonstrates an end-to-end machine learning workflow suitable for environmental analytics and pollution monitoring applications.

## Future Improvements

- Add time-based features such as hour, day, month, and season from `last_update`.
- Include weather variables such as temperature, humidity, wind speed, and rainfall.
- Build classification labels for AQI categories such as Good, Moderate, Poor, and Severe.
- Deploy the trained model using Flask, FastAPI, or Streamlit.
- Add automated model tracking and experiment comparison.

## Resume Highlight

Developed an end-to-end machine learning pipeline to predict real-time air pollutant levels across Indian monitoring stations using Python, Pandas, Scikit-learn, and regression modeling. Performed data cleaning, EDA, feature selection, model benchmarking, and hyperparameter tuning, achieving a best R2 score of 0.967 with Linear Regression.
