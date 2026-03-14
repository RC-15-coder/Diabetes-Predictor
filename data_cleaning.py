import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Load the Dataset
df = pd.read_csv('diabetes.csv')

# Columns where a value of 0 is considered invalid and should be treated as missing
invalid_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[invalid_cols] = df[invalid_cols].replace(0, np.nan)

# Separate Features and Labels
X = df.drop('Outcome', axis=1)  # Features
y = df['Outcome']               # Target variable

# Split the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Combine Training Features and Labels for Preprocessing 
train_data = X_train.copy()
train_data['Outcome'] = y_train

# Conditional Median Imputation
def median_target(column):
    return train_data.groupby('Outcome')[column].median()

for col in invalid_cols:
    # Calculate median values for each class
    medians = median_target(col)
    
    # Impute missing values in training data based on the class
    train_data.loc[(train_data['Outcome'] == 0) & (train_data[col].isnull()), col] = medians[0]
    train_data.loc[(train_data['Outcome'] == 1) & (train_data[col].isnull()), col] = medians[1]
    
    # Impute missing values in test data using training medians
    X_test.loc[(y_test == 0) & (X_test[col].isnull()), col] = medians[0]
    X_test.loc[(y_test == 1) & (X_test[col].isnull()), col] = medians[1]


# Remove Outliers Using Local Outlier Factor (LOF) and initialize LOF with 10 neighbors
lof = LocalOutlierFactor(n_neighbors=10)

# Fit LOF on training data (excluding the target variable)
lof.fit(train_data.drop('Outcome', axis=1))

# Get the negative outlier factor scores
train_scores = lof.negative_outlier_factor_

# Determine the threshold to remove outliers (7th lowest score)
threshold = np.sort(train_scores)[7]

# Keep only the training data points above the threshold
train_data = train_data[train_scores > threshold]

# Update Training and Testing Sets After Outlier Removal
X_train = train_data.drop('Outcome', axis=1)
y_train = train_data['Outcome']

# Reset indices to ensure they are sequential
X_train.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)

scaler = StandardScaler()

# Fit the scaler on the training data and transform both training and test data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the Scaler for Future Use 
scaler_path = os.path.join('predictor', 'ml_model', 'scaler.pkl')
os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
joblib.dump(scaler, scaler_path)

# Save scaled training features
pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv('X_train.csv', index=False)

# Save training labels
pd.DataFrame(y_train, columns=['Outcome']).to_csv('y_train.csv', index=False)

# Save scaled test features
pd.DataFrame(X_test_scaled, columns=X_test.columns).to_csv('X_test.csv', index=False)

# Save test labels
pd.DataFrame(y_test, columns=['Outcome']).to_csv('y_test.csv', index=False)

# Print Summary of Cleaning Process
print("\nMissing values in each column after cleaning (Training Data):")
print(pd.DataFrame(X_train_scaled, columns=X_train.columns).isnull().sum())
print("\nClass distribution in the cleaned training dataset (Outcome counts):")
print(y_train.value_counts())
print("\nTraining dataset shape:", X_train_scaled.shape)
print("Test dataset shape:", X_test_scaled.shape)
