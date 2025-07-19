import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import io

# --- Streamlit App Configuration ---
st.set_page_config(
    page_title="Employee Salary Predictor",
    page_icon="💰",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for better aesthetics ---
st.markdown("""
    <style>
    .main-header {
        font-size: 3em;
        color: #4CAF50;
        text-align: center;
        margin-bottom: 0.5em;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    .subheader {
        font-size: 1.8em;
        color: #333;
        margin-top: 1.5em;
        border-bottom: 2px solid #eee;
        padding-bottom: 0.5em;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        padding: 0.75em 1.5em;
        border-radius: 0.5em;
        border: none;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
        box-shadow: 3px 3px 6px rgba(0,0,0,0.3);
    }
    .stSuccess, .stInfo {
        padding: 1em;
        border-radius: 0.5em;
        margin-top: 1em;
        font-size: 1.1em;
        font-weight: bold;
    }
    .stSuccess {
        background-color: #e6ffe6;
        color: #28a745;
        border: 1px solid #28a745;
    }
    .stInfo {
        background-color: #e0f2f7;
        color: #17a2b8;
        border: 1px solid #17a2b8;
    }
    .stAlert {
        padding: 1em;
        border-radius: 0.5em;
        margin-top: 1em;
        font-size: 1em;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 1. Data Loading and Preprocessing Function (Cached) ---
@st.cache_data(show_spinner="Loading and preprocessing data...")
def load_and_preprocess_data(file_path):
    """Loads the dataset and performs initial cleaning and preprocessing."""
    try:
        data = pd.read_csv(file_path)
        
        # Handle missing values (represented by '?')
        data.replace('?', 'Others', inplace=True)

        # Remove 'Without-pay' and 'Never-worked' from 'workclass'
        data = data[data['workclass'] != 'Without-pay']
        data = data[data['workclass'] != 'Never-worked']

        # Remove specific education categories
        data = data[data['education'] != '1st-4th']
        data = data[data['education'] != '5th-6th']
        data = data[data['education'] != 'Preschool']

        # Drop 'education' column as 'educational-num' provides similar info
        data.drop(columns=['education'], inplace=True)

        # Outlier handling for 'age'
        data = data[(data['age'] <= 75) & (data['age'] >= 17)]

        # Capital-gain outlier handling (cap at 99th percentile)
        upper_bound_cg = data['capital-gain'].quantile(0.99)
        data['capital-gain'] = np.where(data['capital-gain'] > upper_bound_cg, upper_bound_cg, data['capital-gain'])

        # Educational-num outlier handling
        data = data[(data['educational-num'] <= 16) & (data['educational-num'] >= 5)]

        # Convert 'income' to numerical (target variable: 0 for <=50K, 1 for >50K)
        data['income'] = data['income'].apply(lambda x: 1 if x.strip() == '>50K' else 0)

        return data
    except FileNotFoundError:
        st.error(f"Error: The dataset '{file_path}' was not found. Please ensure it's in the same directory as the app.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred during data loading or preprocessing: {e}")
        st.stop()

# --- 2. Model Training Function (Cached) ---
@st.cache_resource(show_spinner="Training the machine learning model...")
def train_prediction_model(X_data, y_data, numerical_features, categorical_features):
    """Trains the KNN model within a preprocessing pipeline."""
    # Create preprocessing pipelines for numerical and categorical features
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore') # handle_unknown='ignore' for robustness

    # Create a column transformer to apply different transformations
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough' # Keep any other columns if they exist
    )

    # Create a pipeline that first preprocesses and then trains the model
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', KNeighborsClassifier(n_neighbors=7)) # Using n_neighbors=7 as determined previously
    ])

    # Split data for internal validation (not used for user input, but good for reporting)
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42, stratify=y_data)

    # Train the model
    model_pipeline.fit(X_train, y_train)

    # Evaluate the model and display in sidebar
    accuracy = model_pipeline.score(X_test, y_test)
    st.sidebar.success(f"Model trained successfully! Accuracy: {accuracy:.2f}")

    return model_pipeline

# --- Main Application Logic ---

# Load and preprocess data
data = load_and_preprocess_data('adult 3.csv')

# Define features (X) and target (y)
X = data.drop('income', axis=1)
y = data['income']

# Identify categorical and numerical columns for preprocessing
# IMPORTANT: Ensure 'gender' is in categorical_cols if it's an object type
categorical_cols = X.select_dtypes(include='object').columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Train the model pipeline
model_pipeline = train_prediction_model(X, y, numerical_cols, categorical_cols)

# --- Streamlit UI ---
st.markdown("<h1 class='main-header'>Employee Salary Predictor 💰</h1>", unsafe_allow_html=True)
st.markdown("""
    Welcome to the Employee Salary Predictor! This application helps you estimate whether an employee's
    annual income is **<=50K** or **>50K** based on their demographic and work-related attributes.
    The prediction is powered by a K-Nearest Neighbors (KNN) classification model.
""")

st.markdown("<h2 class='subheader'>👤 Predict for a Single Employee</h2>", unsafe_allow_html=True)

# Input fields for user
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=17, max_value=75, value=35, help="Age of the employee (17-75 years)")
    workclass = st.selectbox("Workclass", data['workclass'].unique(), help="Type of employer (e.g., Private, Self-emp-not-inc)")
    fnlwgt = st.number_input("Final Weight (fnlwgt)", min_value=10000, value=150000, help="The number of people the census believes the entry represents")
    educational_num = st.slider("Years of Education (educational-num)", min_value=5, max_value=16, value=9, help="Number of years of education completed (5-16)")
    marital_status = st.selectbox("Marital Status", data['marital-status'].unique(), help="Marital status of the employee")
    occupation = st.selectbox("Occupation", data['occupation'].unique(), help="Occupation of the employee")

with col2:
    gender = st.selectbox("Gender", data['gender'].unique(), help="Gender of the employee")
    relationship = st.selectbox("Relationship", data['relationship'].unique(), help="Relationship status (e.g., Husband, Not-in-family)")
    race = st.selectbox("Race", data['race'].unique(), help="Racial background")
    capital_gain = st.number_input("Capital Gain", min_value=0, value=0, help="Capital gains from investments")
    capital_loss = st.number_input("Capital Loss", min_value=0, value=0, help="Capital losses from investments")
    hours_per_week = st.number_input("Hours per Week", min_value=1, max_value=99, value=40, help="Number of hours worked per week")
    native_country = st.selectbox("Native Country", data['native-country'].unique(), help="Country of origin")

# Create a DataFrame from user input
input_data = pd.DataFrame([{
    'age': age,
    'workclass': workclass,
    'fnlwgt': fnlwgt,
    'educational-num': educational_num,
    'marital-status': marital_status,
    'occupation': occupation,
    'relationship': relationship,
    'race': race,
    'gender': gender,
    'capital-gain': capital_gain,
    'capital-loss': capital_loss,
    'hours-per-week': hours_per_week,
    'native-country': native_country
}])

# Prediction button
if st.button("Predict Salary"):
    try:
        # Make prediction
        prediction = model_pipeline.predict(input_data)[0]
        prediction_proba = model_pipeline.predict_proba(input_data)[0]

        st.markdown("### Prediction Result:")
        if prediction == 1:
            st.success(f"The predicted income is: **>50K** 🎉")
        else:
            st.info(f"The predicted income is: **<=50K**")

        st.markdown(f"**Confidence (<=50K):** `{prediction_proba[0]*100:.2f}%`")
        st.markdown(f"**Confidence (>50K):** `{prediction_proba[1]*100:.2f}%`")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.warning("Please ensure all input fields are correctly filled and the dataset is loaded.")

st.markdown("---")

# --- Batch Prediction Section ---
st.markdown("<h2 class='subheader'>📂 Batch Prediction</h2>", unsafe_allow_html=True)
st.markdown("""
    Upload a CSV file containing multiple employee records to get batch salary predictions.
    The uploaded CSV should have the same column names as the input features (e.g., `age`, `workclass`, `fnlwgt`, etc.).
""")

uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type=["csv"])

if uploaded_file is not None:
    try:
        batch_data = pd.read_csv(uploaded_file)
        st.write("Uploaded data preview:")
        st.dataframe(batch_data.head())

        # Check if all required columns are present in the uploaded batch data
        required_cols = X.columns.tolist()
        missing_cols = [col for col in required_cols if col not in batch_data.columns]

        if missing_cols:
            st.error(f"Error: The uploaded CSV is missing the following required columns: `{', '.join(missing_cols)}`")
            st.warning("Please ensure your CSV file has all the necessary feature columns.")
            st.stop()

        # Make batch predictions
        batch_predictions = model_pipeline.predict(batch_data)
        batch_data['Predicted Income'] = np.where(batch_predictions == 1, '>50K', '<=50K')

        st.markdown("### ✅ Predictions for Batch Data:")
        st.dataframe(batch_data.head())

        # Provide download button for predictions
        csv = batch_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Predictions CSV",
            data=csv,
            file_name='predicted_salaries.csv',
            mime='text/csv',
            help="Click to download the CSV with predictions."
        )
    except pd.errors.EmptyDataError:
        st.error("Error: The uploaded CSV file is empty. Please upload a non-empty file.")
    except Exception as e:
        st.error(f"An unexpected error occurred during batch prediction: {e}")
        st.warning("Please ensure your CSV file is correctly formatted and contains valid data.")

st.markdown("---")
st.markdown("© 2025 Internship Project") # Kept this as a standard project footer

# --- Sidebar Content ---
st.sidebar.header("About This App")
st.sidebar.info(
    """
    This application predicts employee salary income using a K-Nearest Neighbors (KNN) classifier.
    It's built using Python and the Streamlit framework.
    """
)
st.sidebar.header("How it Works")
st.sidebar.markdown(
    """
    1.  **Data Loading & Preprocessing:** The `adult_combined.csv` dataset is loaded, cleaned, and preprocessed
        (handling missing values, outliers, and encoding categorical features).
    2.  **Model Training:** A KNN model is trained on the preprocessed data.
    3.  **Prediction:** User inputs (single or batch) are preprocessed using the same pipeline
        and fed to the trained model for salary prediction.
    """
)
st.sidebar.header("Dataset Used")
st.sidebar.markdown(
    """
    The model is trained on a modified version of the Adult Income Dataset,
    which includes various demographic and work-related features.
    """
)
st.sidebar.header("Model Details")
st.sidebar.markdown(
    """
    * **Algorithm:** K-Nearest Neighbors (KNN)
    * **K Value:** 7
    * **Preprocessing:** StandardScaler for numerical features, OneHotEncoder for categorical features.
    """
)
st.sidebar.markdown("---")
st.sidebar.markdown("© 2025 Internship Project")
