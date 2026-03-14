import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve
)
import os
import joblib

# Load the Preprocessed and Scaled Datasets
X_train_path = 'X_train.csv'
y_train_path = 'y_train.csv'
X_test_path = 'X_test.csv'
y_test_path = 'y_test.csv'

# Load the training features and labels
X_train = pd.read_csv(X_train_path)
y_train = pd.read_csv(y_train_path).values.ravel()

# Load the testing features and labels
X_test = pd.read_csv(X_test_path)
y_test = pd.read_csv(y_test_path).values.ravel()

# Initialize the LightGBM Classifier
lgbm_model = LGBMClassifier(
    n_estimators=500,          # Number of boosting rounds
    learning_rate=0.05,        # Step size shrinkage used in update to prevent overfitting
    max_depth=-1,              # No limit on the depth of the tree
    num_leaves=31,             # Maximum tree leaves for base learners
    feature_fraction=0.8,      # Fraction of features to consider at each split
    subsample=0.8,             # Fraction of data to be used for fitting the individual base learners
    subsample_freq=5,          # Frequency of subsampling (0 means no subsampling)
    min_gain_to_split=0.01,    # Minimum gain to make a split
    force_col_wise=True,       # Force column-wise parallel learning
    random_state=42,           # Seed for reproducibility
    verbose=-1                 # Suppress LightGBM logs
)

# Train the LightGBM Model
lgbm_model.fit(X_train, y_train)

# Predict Probabilities on Test Data
y_pred_proba = lgbm_model.predict_proba(X_test)[:, 1]

# Calculate the Optimal Threshold and compute False Positive Rate (FPR), True Positive Rate (TPR), and thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# Determine the threshold where the difference between FPR and TPR is maximized
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print(f"Optimal Threshold: {optimal_threshold:.2f}")

# Make Final Predictions Using the Optimal Threshold
y_pred = (y_pred_proba >= optimal_threshold).astype(int)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Generate classification report (precision, recall, f1-score)
class_report = classification_report(y_test, y_pred)

# Print Evaluation Results
print("\nLightGBM Results:")
print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

# Save the trained model
model_path = os.path.join('predictor', 'ml_model', 'best_lgb_model.pkl')
os.makedirs(os.path.dirname(model_path), exist_ok=True)
joblib.dump(lgbm_model, model_path)






