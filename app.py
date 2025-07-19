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

        # NOTE: 'education' column is now kept as it's an input feature
        # No longer dropping 'education' column here.

        # Outlier handling for 'age'
        data = data[(data['age'] <= 75) & (data['age'] >= 17)]

        # Capital-gain outlier handling (cap at 99th percentile)
        upper_bound_cg = data['capital-gain'].quantile(0.99)
        data['capital-gain'] = np.where(data['capital-gain'] > upper_bound_cg, upper_bound_cg, data['capital-gain'])

        # Educational-num outlier handling (still needed as a feature)
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

# Load and preprocess data (using 'adult 3.csv' as requested)
data = load_and_preprocess_data('adult 3.csv')

# Define features (X) and target (y)
X = data.drop('income', axis=1)
y = data['income']

# Identify categorical and numerical columns for preprocessing
categorical_cols = X.select_dtypes(include='object').columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Train the model pipeline
model_pipeline = train_prediction_model(X, y, numerical_cols, categorical_cols)

# --- Get default values for features not directly inputted by the user ---
# These defaults will be used to fill the full input DataFrame for prediction
default_feature_values = {}
for col in X.columns:
    if col in numerical_cols:
        default_feature_values[col] = data[col].median()
    elif col in categorical_cols:
        default_feature_values[col] = data[col].mode()[0]

# --- Streamlit UI ---
st.markdown("<h1 class='main-header'>Employee Salary Predictor 💰</h1>", unsafe_allow_html=True)
st.markdown("""
    Welcome to the Employee Salary Predictor! This application helps you estimate whether an employee's
    annual income is **<=50K** or **>50K** based on their demographic and work-related attributes.
    The prediction is powered by a K-Nearest Neighbors (KNN) classification model.
""")

st.markdown("<h2 class='subheader'>👤 Enter Employee Details</h2>", unsafe_allow_html=True) # Changed header text

# Input fields for user (only the requested columns)
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=17, max_value=75, value=30, help="Age of the employee (17-75 years)")
    gender = st.selectbox("Gender", data['gender'].unique(), help="Gender of the employee")
    education_level = st.selectbox("Education Level", data['education'].unique(), help="Highest level of education achieved") # Changed to 'education'
    
with col2:
    job_title = st.selectbox("Job Title", data['occupation'].unique(), help="Occupation of the employee") # Changed to 'occupation'
    years_experience = st.number_input("Years of Experience", min_value=0, max_value=50, value=5, help="Years of formal education (maps to educational-num)") # Maps to educational-num

# Create a DataFrame with all columns, initialized with default values
# Then overwrite with user-provided values for the selected inputs
input_data_full = pd.DataFrame([default_feature_values])

# Overwrite with user-provided values
input_data_full['age'] = age
input_data_full['gender'] = gender
input_data_full['education'] = education_level
input_data_full['occupation'] = job_title
input_data_full['educational-num'] = years_experience

# Ensure column order matches X.columns (critical for the model pipeline)
input_data = input_data_full[X.columns]

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
    1.  **Data Loading & Preprocessing:** The `adult 3.csv` dataset is loaded, cleaned, and preprocessed
        (handling missing values, outliers, and encoding categorical features).
    2.  **Model Training:** A KNN model is trained on the preprocessed data.
    3.  **Prediction:** User inputs are taken for key features, and other features are filled with default values.
        The combined data is then fed to the trained model for salary prediction.
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
