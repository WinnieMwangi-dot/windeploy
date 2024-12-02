import pickle
import pandas as pd
import streamlit as st
import os
import traceback
from sklearn.preprocessing import LabelEncoder


# Model file path
model_path = "rf_model_.pkl"  # Update this to the correct path if needed

# Initialize LabelEncoders for categorical features
le_gender = LabelEncoder()
le_education = LabelEncoder()
le_marital_status = LabelEncoder()
le_department = LabelEncoder()
le_job_role = LabelEncoder()
le_travel_freq = LabelEncoder()
le_overtime = LabelEncoder()
le_attrition = LabelEncoder()

# Load the model with error handling
try:
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    st.write(f"Model loaded successfully. Type: {type(model)}")
    
    # Check if the loaded object has a 'predict' method
    if hasattr(model, 'predict'):
        st.write("The model has a 'predict' method and is ready for predictions.")
    else:
        st.error("Loaded object is not a valid model. Please check the file 'rf_model_.pkl'.")
        model = None
except FileNotFoundError:
    st.error(f"Model file '{model_path}' not found. Ensure it exists in the current directory: {os.getcwd()}")
    model = None
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")
    st.error(traceback.format_exc())
    model = None


# Function to get user input for the dataset
def user_input_features():
    # Input features from the sidebar
    Age = st.sidebar.number_input("Age", min_value=18, max_value=65, step=1)
    Gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    EducationBackground = st.sidebar.selectbox("Education Background", ["Science", "Commerce", "Arts", "Others"])
    MaritalStatus = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
    EmpDepartment = st.sidebar.selectbox("Department", ["HR", "Finance", "R&D", "Sales", "IT"])
    EmpJobRole = st.sidebar.selectbox("Job Role", ["Manager", "Executive", "Analyst", "Technician", "Clerk"])
    BusinessTravelFrequency = st.sidebar.selectbox("Business Travel Frequency", ["Rarely", "Frequently", "Never"])
    DistanceFromHome = st.sidebar.number_input("Distance From Home (km)", min_value=0, max_value=100, step=1)
    EmpEducationLevel = st.sidebar.slider("Education Level (1-5)", min_value=1, max_value=5, step=1)
    EmpEnvironmentSatisfaction = st.sidebar.slider("Environment Satisfaction (1-5)", min_value=1, max_value=5, step=1)
    EmpHourlyRate = st.sidebar.number_input("Hourly Rate", min_value=10, max_value=100, step=1)
    EmpJobInvolvement = st.sidebar.slider("Job Involvement (1-5)", min_value=1, max_value=5, step=1)
    EmpJobLevel = st.sidebar.slider("Job Level (1-5)", min_value=1, max_value=5, step=1)
    EmpJobSatisfaction = st.sidebar.slider("Job Satisfaction (1-5)", min_value=1, max_value=5, step=1)
    NumCompaniesWorked = st.sidebar.number_input("Number of Companies Worked", min_value=0, max_value=10, step=1)
    OverTime = st.sidebar.selectbox("Overtime", ["Yes", "No"])
    EmpLastSalaryHikePercent = st.sidebar.number_input("Last Salary Hike Percent", min_value=0, max_value=100, step=1)
    EmpRelationshipSatisfaction = st.sidebar.slider("Relationship Satisfaction (1-5)", min_value=1, max_value=5, step=1)
    TotalWorkExperienceInYears = st.sidebar.number_input("Total Work Experience (Years)", min_value=0, max_value=50, step=1)
    TrainingTimesLastYear = st.sidebar.number_input("Training Times Last Year", min_value=0, max_value=10, step=1)
    EmpWorkLifeBalance = st.sidebar.slider("Work-Life Balance (1-5)", min_value=1, max_value=5, step=1)
    ExperienceYearsAtThisCompany = st.sidebar.number_input("Experience Years At Company", min_value=0, max_value=50, step=1)
    ExperienceYearsInCurrentRole = st.sidebar.number_input("Experience Years In Current Role", min_value=0, max_value=50, step=1)
    YearsSinceLastPromotion = st.sidebar.number_input("Years Since Last Promotion", min_value=0, max_value=50, step=1)
    YearsWithCurrManager = st.sidebar.number_input("Years With Current Manager", min_value=0, max_value=50, step=1)
    Attrition = st.sidebar.selectbox("Attrition", ["Yes", "No"])

    # Apply Label Encoding to categorical features
    Gender = le_gender.fit_transform([Gender])[0]
    EducationBackground = le_education.fit_transform([EducationBackground])[0]
    MaritalStatus = le_marital_status.fit_transform([MaritalStatus])[0]
    EmpDepartment = le_department.fit_transform([EmpDepartment])[0]
    EmpJobRole = le_job_role.fit_transform([EmpJobRole])[0]
    BusinessTravelFrequency = le_travel_freq.fit_transform([BusinessTravelFrequency])[0]
    OverTime = le_overtime.fit_transform([OverTime])[0]
    Attrition = le_attrition.fit_transform([Attrition])[0]

    # Combine inputs into a dataframe
    data = {
        'Age': Age,
        'Gender': Gender,
        'EducationBackground': EducationBackground,
        'MaritalStatus': MaritalStatus,
        'EmpDepartment': EmpDepartment,
        'EmpJobRole': EmpJobRole,
        'BusinessTravelFrequency': BusinessTravelFrequency,
        'DistanceFromHome': DistanceFromHome,
        'EmpEducationLevel': EmpEducationLevel,
        'EmpEnvironmentSatisfaction': EmpEnvironmentSatisfaction,
        'EmpHourlyRate': EmpHourlyRate,
        'EmpJobInvolvement': EmpJobInvolvement,
        'EmpJobLevel': EmpJobLevel,
        'EmpJobSatisfaction': EmpJobSatisfaction,
        'NumCompaniesWorked': NumCompaniesWorked,
        'OverTime': OverTime,
        'EmpLastSalaryHikePercent': EmpLastSalaryHikePercent,
        'EmpRelationshipSatisfaction': EmpRelationshipSatisfaction,
        'TotalWorkExperienceInYears': TotalWorkExperienceInYears,
        'TrainingTimesLastYear': TrainingTimesLastYear,
        'EmpWorkLifeBalance': EmpWorkLifeBalance,
        'ExperienceYearsAtThisCompany': ExperienceYearsAtThisCompany,
        'ExperienceYearsInCurrentRole': ExperienceYearsInCurrentRole,
        'YearsSinceLastPromotion': YearsSinceLastPromotion,
        'YearsWithCurrManager': YearsWithCurrManager,
        'Attrition': Attrition
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Ensure the model is loaded before proceeding
if model is not None:
    # Load user input
    inx = user_input_features()

    # Display input
    st.subheader("User Input:")
    st.write(inx)

    # Make prediction
    if st.button("Predict"):
        try:
            prediction = model.predict(inx)
            st.subheader("Prediction: ")
            st.write(prediction[0])  # Display the predicted value
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.error(traceback.format_exc())
else:
    st.error("Model is not loaded. Please check the file path or loading process.")
