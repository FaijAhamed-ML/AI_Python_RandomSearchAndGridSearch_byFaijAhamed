from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split , GridSearchCV , RandomizedSearchCV # for hyperparameter tuning
from sklearn.ensemble import RandomForestClassifier # this is for classification tasks(use RandomForestRegressor for regression tasks)
from sklearn.metrics import accuracy_score
import numpy as np

# Load dataset
data = load_iris()
X = data.data
y = data.target
# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#display dataset info 
print(f"Feature names: {data.feature_names}")
print(f"Target names: {data.target_names}")

#define hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 150], # this is for number of trees in the forest
    'max_depth': [None, 5,10], # this is for maximum depth of the tree
    'min_samples_split': [2, 5, 10], # this is for minimum number of samples required to split an internal node
}

#initialize grid search
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                           param_grid=param_grid,
                           cv=5, # number of cross-validation folds
                           scoring='accuracy',
                           n_jobs=-1, # use all available cores
)

#perform grid search
grid_search.fit(X_train, y_train) # fit the model(training)

#Evaluate the best model
best_grid_model = grid_search.best_estimator_ # get the best model from grid search
y_pred_grid = best_grid_model.predict(X_test) # make predictions on the test set
accuracy_grid = accuracy_score(y_test, y_pred_grid) # calculate accuracy
print(f"Best Grid Search Model Accuracy: {accuracy_grid:.4f}") 

#display best hyperparameters
print("Best Hyperparameters from Grid Search:")
print(grid_search.best_params_)

#define hyperparameter distribution for random search
param_dist = {
    'n_estimators': np.arange(50, 200, 10), # this is for number of trees in the forest
    'max_depth': [None, 5,10,15], # this is for maximum depth of the tree
    'min_samples_split': [2, 5, 10,20], # this is for minimum number of samples required to split an internal node
}

#initialize random search
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42), # this is for classification tasks(use RandomForestRegressor for regression tasks)
    param_distributions=param_dist, # hyperparameter distributions
    n_iter=20, # number of random combinations to try
    cv=5, # number of cross-validation folds
    scoring='accuracy',
    n_jobs=-1, # use all available cores
)

#perform random search
random_search.fit(X_train, y_train) # fit the model(training)

#Evaluate the best model
best_random_model = random_search.best_estimator_ # get the best model from random search
y_pred_random = best_random_model.predict(X_test) # make predictions on the test set
accuracy_random = accuracy_score(y_test, y_pred_random) # calculate accuracy
print(f"Best Random Search Model Accuracy: {accuracy_random:.4f}")
#display best hyperparameters
print("Best Hyperparameters from Random Search:")
print(random_search.best_params_) 