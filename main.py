import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Function to predict heart disease
def predict_heart_disease(user_input, X_train, y_train, feature_order):
    # Create DataFrame from user input, ensuring columns are in the correct order
    X = pd.DataFrame(user_input, index=[0])
    X = X[feature_order]  # Align columns to match the order used during training
    
    # Standardizing the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_scaled = scaler.transform(X)
    
    # Initialize and train the MLPClassifier
    mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000, random_state=42)
    
    # Train the model
    mlp.fit(X_train_scaled, y_train)
    
    # Predict based on user input
    prediction = mlp.predict(X_scaled)[0]
    
    return 'Yes' if prediction == 1 else 'No'

# Load the dataset
uploaded_file = "data.csv"
df = pd.read_csv(uploaded_file)

# Standardize column names (ensure they match the user inputs)
df.rename(columns={'trestbps': 'thestbps'}, inplace=True)

# Prepare data for training
X = df.drop(columns=['num'])  # Ensure 'num' is the correct target column
y = df['num']  # Ensure 'num' is the correct target column

# Get the order of features from the training set
feature_order = X.columns.tolist()

# Split the data into training and testing sets
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=42)

# Streamlit interface
st.title('Heart Disease Prediction System')
st.markdown("""
**Important Notice:**
Before entering your values, please ensure that you have recently undergone a complete blood test. Accurate and up-to-date medical data is crucial for reliable predictions. We recommend consulting with your healthcare provider to obtain the necessary test results before using this system.
""")

# Input fields for user to provide data
age = st.number_input('Age', min_value=1, max_value=120, value=30, help="Enter the age in years.")
sex = st.selectbox('Sex', options=['Male', 'Female'], help="Select the biological sex of the patient.")
cp = st.selectbox('Chest Pain Type', options=['Typical angina', 'Atypical angina', 'Non-anginal pain', 'Asymptomatic'],
                  help="Type of chest pain experienced.")
thestbps = st.number_input('Resting Blood Pressure (mm Hg)', min_value=50, max_value=200, value=120, 
                           help="The resting blood pressure in mm Hg.")  # Updated key to match dataset
chol = st.number_input('Serum Cholesterol (mg/dL)', min_value=100, max_value=600, value=200, 
                       help="Serum cholesterol level in mg/dL.")
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dL', options=['Yes', 'No'], help="Is fasting blood sugar greater than 120 mg/dL?")
restecg = st.selectbox('Resting ECG Results', options=['Normal', 'Having ST-T wave abnormality', 'Left ventricular hypertrophy'],
                       help="Resting electrocardiographic results.")
thalach = st.number_input('Maximum Heart Rate Achieved', min_value=50, max_value=250, value=150, 
                          help="Maximum heart rate achieved during test.")
exang = st.selectbox('Exercise Induced Angina', options=['Yes', 'No'], help="Is there angina induced by exercise?")
oldpeak = st.number_input('ST Depression Induced by Exercise Relative to Rest', min_value=0.0, max_value=10.0, value=1.0, 
                          help="ST depression induced by exercise relative to rest.")
slope = st.selectbox('Slope of the Peak Exercise ST Segment', options=['Upsloping', 'Flat', 'Downsloping'],
                     help="The slope of the peak exercise ST segment.")
ca = st.selectbox('Number of Major Vessels Colored by Fluoroscopy', options=[0, 1, 2, 3],
                  help="Number of major vessels colored by fluoroscopy.")
thal = st.selectbox('Thalassemia', options=['Normal', 'Fixed Defect', 'Reversible Defect'],
                    help="Thalassemia result.")
obes = st.selectbox('Obesity', options=['Yes', 'No'], help="Is the patient obese?")

# Prepare user input for prediction
user_input = {
    'age': age,
    'sex': 1 if sex == 'Male' else 0,
    'cp': ['Typical angina', 'Atypical angina', 'Non-anginal pain', 'Asymptomatic'].index(cp) + 1,
    'thestbps': thestbps,  # Ensure this key matches the training data
    'chol': chol,
    'fbs': 1 if fbs == 'Yes' else 0,
    'restecg': ['Normal', 'Having ST-T wave abnormality', 'Left ventricular hypertrophy'].index(restecg),
    'thalach': thalach,
    'exang': 1 if exang == 'Yes' else 0,
    'oldpeak': oldpeak,
    'slope': ['Upsloping', 'Flat', 'Downsloping'].index(slope) + 1,
    'ca': ca,
    'thal': ['Normal', 'Fixed Defect', 'Reversible Defect'].index(thal) + 3,
    'obes': 1 if obes == 'Yes' else 0
}

# Predict heart disease based on input
if st.button('Predict'):
    prediction = predict_heart_disease(user_input, X_train, y_train, feature_order)
    st.write(f"Prediction: Heart Disease - {prediction}")
