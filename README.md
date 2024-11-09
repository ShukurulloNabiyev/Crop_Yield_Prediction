
# Crop Yield Prediction and Analysis

This repository contains code and documentation for analyzing and predicting crop yield using various environmental, climatic, and pollinator-related features. A `RandomForestRegressor` model is used for predictions, and SHAP (SHapley Additive exPlanations) is employed for feature importance analysis. Additionally, the repository provides statistical and exploratory analyses of the features.

## Project Structure

- `best_model copy.ipynb`: Jupyter notebook containing data preprocessing, model training, evaluation, and SHAP-based feature importance analysis.
- `analaysis.ipynb`: Jupyter notebook for exploratory data analysis (EDA) and statistical analysis of the dataset.
- `README.md`: Project documentation (this file).

## Requirements

To run the code in both notebooks, install the necessary libraries by executing:

```bash
pip install pandas numpy scikit-learn shap matplotlib
```

## Dataset

The dataset is expected to contain the following types of features:

- **Pollinator Features**: Includes features like `osmia`, `honeybee`, and `clonesize`, which represent the presence and activity of various pollinators.
- **Climate Features**: Features like `RainingDays`, `AverageRainingDays`, `MaxOfUpperTRange`, `MinOfUpperTRange`, `AverageOfUpperTRange`, `MaxOfLowerTRange`, and `MinOfLowerTRange`, which capture climatic conditions.
- **Yield and Related Metrics**: The target variable `yield`, along with other yield-related features like `fruitset`, `seeds`, and `fruitmass`.

Some columns, including `clonesize`, `honeybee`, `AverageOfUpperTRange`, `MaxOfLowerTRange`, `MinOfUpperTRange`, and `MinOfLowerTRange`, are excluded from the model input due to lower relevance.

## Data Analysis (Exploratory and Statistical)

The `analaysis.ipynb` notebook performs an exploratory data analysis to uncover patterns and relationships among features. Key analysis includes:

- **Feature Distributions**: Visualization of individual feature distributions to understand the data spread.
- **Correlation Analysis**: Analysis of correlations between features and the target variable (`yield`).
- **Eta Squared Calculation**: Calculation of effect size (eta squared) for various features to quantify their relationship with the yield.

## Model Training and Evaluation

The `RandomForestRegressor` model is trained with specific hyperparameters:

- `n_estimators=400`: Number of trees in the forest.
- `max_depth=11`: Maximum depth of each tree.
- `criterion='absolute_error'`: Criterion for measuring the quality of a split.
- `min_samples_leaf=5`: Minimum number of samples required in a leaf node.
- `max_features='log2'`: Number of features considered when looking for the best split.
- `min_impurity_decrease=0.065`: Minimum impurity decrease required for a split.

### Code Example

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Splitting the data
X = df.drop(columns=['clonesize', 'honeybee', 'AverageOfUpperTRange', 'MaxOfLowerTRange', 'MinOfUpperTRange', 'MinOfLowerTRange', 'yield'])
y = df['yield']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the model
rf = RandomForestRegressor(random_state=42, n_estimators=400, max_depth=11, criterion='absolute_error', min_samples_leaf=5, max_features='log2', min_impurity_decrease=0.065)
rf.fit(X_train, y_train)

# Evaluating the model
y_pred = rf.predict(X_test)
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))
```

### Model Evaluation Metrics

- **Mean Absolute Error (MAE)**: Measures the average absolute difference between predicted and actual values.
- **R² Score**: Indicates how well the model's predictions approximate the actual values.

## Feature Engineering

Various new features were engineered based on the original dataset to improve the model’s performance. Examples include:

1. **Pollinator Interactions**: Created interaction terms like `osmia_honeybee_interaction` and `pollinators_clonesize_interaction`.
2. **Rain-Pollinator Combined Features**: Added features like `rain_osmia` and `rain_honeybee` to capture the impact of rainfall on pollinators.
3. **Fruitset Ratios**: Computed ratios such as `seed_fruitmass_ratio` and `fruitset_per_osmia`.
4. **Discretization**: Discretized continuous features like `fruitmass` and `RainingDays` to capture non-linear patterns.

## Feature Importance with SHAP

SHAP values are used to interpret the feature importance in the model. The summary plots help in understanding the contributions of individual features.

### SHAP Code Example

```python
import shap
import matplotlib.pyplot as plt

# Initialize the SHAP explainer with the trained model
explainer = shap.Explainer(rf, X_train)
shap_values = explainer(X_test)

# Summary plot of feature importances
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, plot_type="bar")
plt.title("Feature Importance (SHAP Values)")

# Summary plot to show distribution of SHAP values
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test)
plt.title("SHAP Summary Plot")
plt.show()
```

## Running the Notebooks

To execute the notebooks:

1. Open `best_model copy.ipynb` and `analaysis.ipynb` in Jupyter Notebook or JupyterLab.
2. Run each cell sequentially to perform data analysis, train the model, evaluate its performance, and interpret feature importance.

## License

This project is licensed under the MIT License.
