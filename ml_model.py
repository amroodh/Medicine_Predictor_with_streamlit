# ml_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
import joblib

# Load the dataset
file_path = 'Dataset.csv'  # Adjust the path if necessary
data = pd.read_csv(file_path)

# Data Preprocessing
label_encoders = {}
for column in data.columns:
    if data[column].dtype == 'object' and column not in ['Medicine', 'Dosage']:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column].astype(str))
        label_encoders[column] = le

# Handle missing values
data = data.dropna()

# Split data into features and target variables
X = data.drop(columns=['Medicine', 'Dosage'])
y_medicine = data['Medicine']
y_dosage = data['Dosage']

# Encode 'Medicine' and 'Dosage'
le_medicine = LabelEncoder()
y_medicine = le_medicine.fit_transform(y_medicine)
label_encoders['Medicine'] = le_medicine

le_dosage = LabelEncoder()
y_dosage = le_dosage.fit_transform(y_dosage)
label_encoders['Dosage'] = le_dosage

# Split data into training and testing sets
X_train, X_test, y_train_medicine, y_test_medicine = train_test_split(X, y_medicine, test_size=0.2, random_state=42)
_, _, y_train_dosage, y_test_dosage = train_test_split(X, y_dosage, test_size=0.2, random_state=42)

# Build the MLP model pipeline for medicine prediction
mlp_pipeline_medicine = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=1000, random_state=42))
])

# Train the medicine predictor
mlp_pipeline_medicine.fit(X_train, y_train_medicine)

# Save the medicine model and label encoders
joblib.dump(mlp_pipeline_medicine, 'medicine_predictor.pkl')

# Build the MLP model pipeline for dosage prediction
mlp_pipeline_dosage = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=1000, random_state=42))
])

# Train the dosage predictor
mlp_pipeline_dosage.fit(X_train, y_train_dosage)

# Save the dosage model and label encoders
joblib.dump(mlp_pipeline_dosage, 'dosage_predictor.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
