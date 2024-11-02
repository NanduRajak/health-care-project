import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV

# Load the dataset
def load_data(file_path):
    """Load dataset from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        print("Dataset loaded successfully.")
        
        # Check for required columns and add missing columns with synthetic data
        if 'SurvivalDays' not in data.columns:
            print("SurvivalDays column not found. Adding synthetic data for SurvivalDays.")
            data['SurvivalDays'] = np.random.randint(100, 1000, size=len(data))  # Generate random survival days

        if 'HealthCondition' not in data.columns:
            print("HealthCondition column not found. Adding synthetic health condition labels.")
            data['HealthCondition'] = np.random.choice([0, 1], size=len(data))  # Random 0 or 1 for health condition
        
        return data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

# Preprocess the data
def preprocess_data(data):
    """Preprocess data: handle categorical values and standardize numerical values."""
    # Encode categorical variables if they exist
    label_encoders = {}
    categorical_columns = ['Gender', 'FamilyHistory', 'SmokingStatus']

    for column in categorical_columns:
        if column in data.columns:
            le = LabelEncoder()
            data[column] = le.fit_transform(data[column])
            label_encoders[column] = le  # Save encoder if needed for inverse transformation
        else:
            print(f"Warning: '{column}' column not found.")
    
    # Drop irrelevant columns or target columns
    X = data.drop(columns=[col for col in ['HealthCondition', 'SurvivalDays', 'ID'] if col in data.columns])
    y = data[['HealthCondition', 'SurvivalDays']]
    
    # Standardize the numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler

# Train models for classification and survival prediction
def train_models(X_train, y_train_class, y_train_survival):
    """Train a classifier for HealthCondition and a regressor for SurvivalDays."""
    # Classifier for health risk prediction
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train, y_train_class)

    # Hyperparameter tuning for the regressor
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10]
    }
    
    # Regressor for survival prediction
    regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    grid_search = GridSearchCV(regressor, param_grid, cv=3, scoring='neg_mean_absolute_error')
    grid_search.fit(X_train, y_train_survival)

    # Best estimator from grid search
    tuned_regressor = grid_search.best_estimator_
    
    print(f"Best parameters for regressor: {grid_search.best_params_}")
    print(f"Best MAE from training: {-grid_search.best_score_}")
    
    return classifier, tuned_regressor

# Evaluate the classifier model
def evaluate_classifier(classifier, X_test, y_test_class):
    """Evaluate c
    lassifier performance using test data."""
    y_pred = classifier.predict(X_test)
    print("Classification Model Accuracy:", accuracy_score(y_test_class, y_pred))
    print("\nClassification Report:\n", classification_report(y_test_class, y_pred))
    
    # Set all possible labels in the confusion matrix
    labels = [0, 1]  # Assuming 0 = Low risk and 1 = High risk
    print("\nConfusion Matrix:\n", confusion_matrix(y_test_class, y_pred, labels=labels))

# Example evaluate_regressor function to calculate MAE
def evaluate_regressor(regressor, X_test, y_test_survival):
    """Evaluate regressor performance using test data."""
    y_pred = regressor.predict(X_test)
    mae = mean_absolute_error(y_test_survival, y_pred)
    print("Mean Absolute Error (MAE) for Survival Prediction:", mae)
    
# Main function to run the pipeline
def main():
    # Load data
    file_path = "health_data.csv.rtf"
    data = load_data(file_path)
    if data is None:
        return
    
    # Preprocess data
    X, y, scaler = preprocess_data(data)
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Separate targets for classification and regression
    y_train_class = y_train['HealthCondition']
    y_train_survival = y_train['SurvivalDays']
    y_test_class = y_test['HealthCondition']
    y_test_survival = y_test['SurvivalDays']
    
    # Train models
    classifier, regressor = train_models(X_train, y_train_class, y_train_survival)
    
    # Evaluate models
    print("\n--- Classification Model Evaluation ---")
    evaluate_classifier(classifier, X_test, y_test_class)
    
    print("\n--- Regression Model Evaluation ---")
    evaluate_regressor(regressor, X_test, y_test_survival)

if __name__ == "__main__":
    main()
