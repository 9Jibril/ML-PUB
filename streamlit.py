import streamlit as st
import pandas as pd
import joblib

# Load the trained model and encoders
model_filename = 'best_model.pkl'  # Replace with your model file path
encoders_filename = 'encoders.pkl'  # Replace with your encoders file path

try:
    model = joblib.load(model_filename)
    encoders = joblib.load(encoders_filename)
except FileNotFoundError:
    st.error("Model or encoders not found. Ensure 'best_model.pkl' and 'encoders.pkl' are in the correct directory.")

# Function to display "About" information with enhanced styling
def display_about_info():
    st.subheader("About")
    st.markdown(
        "<div style='font-size: 18px; font-weight: bold; color: #001F3F;'>"
        "Welcome to the Machine Learning Predictions system. "
        "This platform is designed to help Educational Technology Students’ in predicting their academic performance. "
        "It allows users to input information and predicts their final CGPA based on a trained SVM model. "
        "This system also displays the corresponding class of degree based on the predicted CGPA . "
        "Additionally, it also provides students with their current calculated GPA and Class of Degree. "
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size: 16px; font-weight: bold; color: #008000;'>"
        "<b>How it Works:</b><br>"
        "1. <i>Input Features:</i> Enter compulsory features and select additional features as needed.<br>"
        "2. <i>Calculate CGPA:</i> The app calculates the CGPA based on user input for courses.<br>"
        "3. <i>Determine Class of Degree:</i> The app determines the Class of Degree from the calculated CGPA.<br>"
        "4. <i>Model Prediction:</i> The system then predict the final CGPA and display corresponding class of degree using a trained SVM model."
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size: 16px; font-weight: bold; color: #FF0000;'>"
        "<b>Note:</b><br>"
        "This system is developed for educational purposes only. While it offers valuable predictions, users are advised to ensure correct input and refer to the model's limitations. The application's primary goal is to provide users with insights and a tool for academic planning. "
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
    f'<h1 style="{title_style}">Educational Technology Students Academic Performance Predictor</h1>',
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

# Sidebar for user input
st.sidebar.header("Enter Student Details")

# Dropdown for Course
course = st.sidebar.selectbox(
    "Select Course",
    options=['Select an Option', 'EDU 214', 'EDU 414']
)

# Dropdown for Gender
gender = st.sidebar.selectbox(
    "Select Gender",
    options=['Select an Option', 'Male', 'Female']
)

# Dropdown for State
state = st.sidebar.selectbox(
    "Select State",
    options=[
        'Select an Option', 'kwara', 'lagos', 'ondo', 'osun', 'ekiti', 'oyo', 'kogi', 'ogun', 'benue', 'imo',
        'cross river', 'delta', 'edo', 'abia', 'enugu', 'anambra', 'kaduna', 'akwa ibom',
        'kano', 'plateau', 'ebonyi', 'bayelsa'
    ]
)

# Dropdown for Mode of Entry
mode_of_entry = st.sidebar.selectbox(
    "Select Mode of Entry",
    options=['Select an Option', 'UTME', 'Direct', 'Remedial']
)

# Number input for Age
age = st.sidebar.number_input(
    "Enter Age (15 - 40)",
    min_value=15,
    max_value=40,
    step=1
)

# Number input for Test Score
test_score = st.sidebar.number_input(
    "Enter Test Score (0 - 40)",
    min_value=0,
    max_value=40,
    step=1
)

# Validation checks
valid_inputs = (
    course != 'Select an Option' and
    gender != 'Select an Option' and
    state != 'Select an Option' and
    mode_of_entry != 'Select an Option' and
    15 <= age <= 40 and
    0 <= test_score <= 40
)

# Display warning if inputs are invalid
if not valid_inputs:
    st.sidebar.warning("Ensure all fields are filled correctly and Age/Test Score are within valid ranges.")

# Button to trigger prediction
if st.sidebar.button("Predict Exam Score"):
    if valid_inputs:
        # Encode categorical inputs
        encoded_course = encoders['Course'].transform([course])
        encoded_gender = encoders['Gender'].transform([gender])
        encoded_state = encoders['State'].transform([state])
        encoded_mode_of_entry = encoders['Mode of Entry'].transform([mode_of_entry])

        # Prepare the data for prediction
        input_data = pd.DataFrame({
            'Course': [encoded_course[0]],
            'Gender': [encoded_gender[0]],
            'State': [encoded_state[0]],
            'Mode of Entry': [encoded_mode_of_entry[0]],
            'Age': [age],
            'Test': [test_score]
        })

        # Perform prediction
        try:
            prediction = model.predict(input_data)
            st.markdown(f"<h2 class='prediction-output'>The predicted exam score is: {prediction[0]:.2f}</h2>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Prediction Error: {e}")
    else:
        # No prediction if inputs are invalid
        st.error("Cannot make a prediction. Please ensure all fields are valid and filled correctly.")
