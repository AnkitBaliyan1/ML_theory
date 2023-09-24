Certainly! Here's how you can perform hyperparameter tuning using RandomizedSearchCV with a RandomForestClassifier:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

# Define the parameter grid for tuning
param_dist = {
    'n_estimators': [100, 200, 300],  # Number of trees in the forest
    'max_depth': [None, 5, 10, 15],  # Maximum depth of the trees
    'criterion': ['gini', 'entropy'],  # Split criterion
    'random_state': [99]  # Random seed for reproducibility
}

# Create the RandomForestClassifier
rf_classifier = RandomForestClassifier()

# Create RandomizedSearchCV with 5-fold cross-validation
random_search = RandomizedSearchCV(
    estimator=rf_classifier, param_distributions=param_dist,
    n_iter=10, cv=5, n_jobs=-1, random_state=42)

# Fit the randomized search to the data
random_search.fit(X, y)  # Replace X and y with your data

# Print the best hyperparameters found by RandomizedSearchCV
print("Best Parameters:", random_search.best_params_)

# Print the best cross-validation score
print("Best Cross-Validation Score:", random_search.best_score_)
```

In this code:

- We use `RandomizedSearchCV` instead of `GridSearchCV`.
- The `param_distributions` parameter is used to specify the hyperparameter space to sample from.
- `n_iter` controls the number of random combinations to try, which is typically smaller than the total number of combinations in a grid search.

The rest of the code is similar to the grid search example. It will randomly sample hyperparameter combinations from the specified space and evaluate them using cross-validation, ultimately selecting the best combination based on performance.