import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the saved model and label encoders
model = joblib.load('best_hist_gradient_boosting_model.pkl')
label_encoders = joblib.load('LabelEncoders.pkl')

# Function to display "About" information with enhanced styling
def display_about_info():
    st.subheader("About")
    st.markdown(
        "<div style='font-size: 18px; font-weight: bold; color: #001F3F;'>"
        "Welcome to the Machine Learning Predictions system. "
        "The Educational Technology Academic Performance Prediction System is an advanced tool designed to predict students' exam scores within the Faculty of Education. "
        "The system specifically focuses on the core educational technology courses that are offered to all students across various educational programs within the faculty. "
        "This predictive system leverages a HistGradientBoosting Regressor, which has been trained using a comprehensive dataset of student performance data sourced from various Nigerian universities. "
        "This system assists in identifying students who may need additional support, enabling educators to make well-informed decisions that can enhance students' academic performance. "
        "</div>",
        unsafe_allow_html=True,
    
    )
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size: 16px; font-weight: bold; color: #FF0000;'>"
        "<b>Note:</b><br>"
        "This system is developed for educational purposes only. Ensure all input data is accurate for reliable predictions. The system's predictions are based on historical data and may not always reflect future performance. Results are for guidance and should not replace professional assessments. Always consider other factors when evaluating student performance. "
        " If you have any concerns or inquiries, please don't hesitate to contact 9jibrilmohammed@email.com. Thank you for your understanding and for using the Machine Learning Predictions system! "
        "</div>",
        unsafe_allow_html=True,
    )

# Custom CSS for title with deep blue background
title_style = (
    "color: white; "
    "background-color: #001F3F; "  # Deep Blue Color
    "padding: 10px; "
    "border-radius: 10px;"
)

# Apply the custom CSS to the title
st.markdown(
    f'<h1 style="{title_style}">Educational Technology Academic Performance Prediction System</h1>',
    unsafe_allow_html=True,
)

# Expander for "Preface" information
with st.expander("Preface"):
    display_about_info()

# CSS for black and white theme, and blue emphasis for "About" section
st.markdown("""
    <style>
        body {
            color: black;
            background-color: white;
        }
        .dropdown-message {
            background-color: #003366;
            color: white;
            padding: 15px;
            border-radius: 10px;
            font-size: 1rem;
        }
        .stButton button {
            color: black;
            background-color: white;
            border: 1px solid black;
            font-size: 16px;
            padding: 8px 16px;
            border-radius: 8px;
        }
        .stButton button:hover {
            background-color: #f5f5f5;
        }
        .stAlert {
            color: black;
        }
        .stSuccess {
            color: black;
        }
        .stError {
            color: black;
        }
        .stMarkdown {
            color: black;
        }
        .stText {
            color: black;
        }
        .prediction-output {
            font-size: 2rem;
            font-weight: bold;
            color: black;
        }
    </style>
    """, unsafe_allow_html=True)

# Function to encode user input data using the saved label encoders
def encode_input(data, label_encoders):
    for column in ['Course', 'Gender', 'State', 'Mode of Entry']:
        le = label_encoders[column]
        data[column] = le.transform([data[column]])[0]
    return data

# Input fields for the user
st.header('Enter Student Information')

course = st.selectbox('Course', ['EDU 214', 'EDU 414'])  # Add your actual course names here
gender = st.selectbox('Gender', ['Male', 'Female'])
state = st.selectbox('State', ['benue', 'delta', 'edo', 'ekiti', 'kaduna', 'kano', 'kogi', 'kwara', 'lagos', 'ogun', 'ondo', 'osun', 'oyo'])  # List of all unique states
mode_of_entry = st.selectbox('Mode of Entry', ['Direct', 'Remedial', 'UTME'])  # List of all modes
age = st.number_input('Age', min_value=18, max_value=100, value=20)
test_score = st.number_input('Test Score', min_value=0, max_value=40, value=40)

# Prepare the input data as a DataFrame
input_data = pd.DataFrame({
    'Course': [course],
    'Gender': [gender],
    'State': [state],
    'Mode of Entry': [mode_of_entry],
    'Age': [age],
    'Test': [test_score]
})

# Button to make prediction
if st.button('Predict Exam Score'):
    # Encode the input data
    encoded_data = encode_input(input_data.copy(), label_encoders)

    # Make prediction using the trained model
    predicted_exam_score = model.predict(encoded_data)[0]

    # Display the predicted result
    st.subheader(f"Predicted Exam Score: {predicted_exam_score:.2f}")

# Optional: Display the raw values as a table
st.subheader("Input Data")
st.write(input_data)

