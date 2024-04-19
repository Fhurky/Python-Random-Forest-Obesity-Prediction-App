from time import sleep
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np

# Read the CSV file
with open("Data_set.csv", "r") as file:
    lines = file.readlines()

# Convert each line into a list by splitting at commas
data = [line.strip().split(",") for line in lines]

# Create the DataFrame
df = pd.DataFrame(data, columns=["Age", "Gender", "Height", "Weight", "CALC", "FAVC", "FCVC", "NCP", "SCC", "SMOKE", "CH2O", "family_history_with_overweight", "FAF", "TUE", "CAEC", "MTRANS", "NObeyesdad"])

# Explore the dataset and perform preprocessing
# For example, you can fill missing values, encode categorical variables, etc.

# Separate the target variable (NObeyesdad) and features
X = df.drop("NObeyesdad", axis=1)
y = df["NObeyesdad"]

# Encode categorical variables
label_encoder = LabelEncoder()
X_encoded = X.apply(label_encoder.fit_transform)

# Choose and train the model
model = RandomForestClassifier(n_estimators=4, random_state=42)
model.fit(X_encoded, y)

# Evaluate the model's performance
y_pred = model.predict(X_encoded)
accuracy = accuracy_score(y, y_pred)
report = classification_report(y, y_pred)

print("Modelin Performansı:")
print("Accuracy:", accuracy)
print("Classification Report:")
print(report)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Evaluate the model's performance on the testing set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Model Performance:")
print("Accuracy:", accuracy)
print("Classification Report:")
print(report)

# Get predictions for the first 10 individuals in the dataset
first_10_predictions = model.predict(X_encoded.head(10))

print("\nPredictions for the First 10 Individuals in the Training Set:")
for i, prediction in enumerate(first_10_predictions):
    print(f"Individual {i + 1} Prediction: {prediction}")

# Create a random person
new_person = pd.DataFrame({
    "Age": [23],  # Sample age
    "Gender": ["Male"],  # Sample gender
    "Height": [1.80],  # Sample height
    "Weight": [75],  # Sample weight (indicating they are overweight)
    "CALC": ["no"],  # Sample value
    "FAVC": ["no"],  # Sample value
    "FCVC": [2],  # Sample value
    "NCP": [3],  # Sample value
    "SCC": ["no"],  # Sample value
    "SMOKE": ["no"],  # Sample value
    "CH2O": [2],  # Sample value
    "family_history_with_overweight": ["yes"],  # Sample value
    "FAF": [0],  # Sample value
    "TUE": [1],  # Sample value
    "CAEC": ["Sometimes"],  # Sample value
    "MTRANS": ["Public_Transportation"]  # Sample value
})

# Get the classes of categorical variables in the training data
label_encoder_classes = {}
for col in X.columns:
    if X[col].dtype == 'object':
        label_encoder = LabelEncoder()
        label_encoder.fit(X[col])
        label_encoder_classes[col] = label_encoder.classes_

# Encode the new person's data using the label_encoder_classes
new_person_encoded = new_person.copy()
for col, classes in label_encoder_classes.items():
    new_person_encoded[col] = new_person[col].apply(lambda x: np.where(classes == x)[0][0] if x in classes else -1)

# Get the prediction for the new person
prediction = model.predict(new_person_encoded)

# Print the predicted result
print("\nPredicted Result:", prediction)