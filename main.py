import pandas as pd
import streamlit as st
import joblib

# Set page configuration at the top of the script
st.set_page_config(page_title="Churn Prediction App", layout="centered")

# Load the saved model
MODEL_PATH = "Model/logistic_regression_pipeline.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model_pipeline = load_model()

# Streamlit app
def main():
    st.title("Customer Churn Prediction")

    st.write("Enter customer details to predict churn:")

    # User input form
    with st.form("churn_prediction_form"):
        state = st.text_input("State (e.g., NY):", max_chars=2)
        account_length = st.number_input("Account Length (e.g., 120):", min_value=0, step=1)
        area_code = st.number_input("Area Code (e.g., 415):", min_value=100, max_value=999, step=1)
        international_plan = st.selectbox("International Plan:", ["Yes", "No"])
        voice_mail_plan = st.selectbox("Voice Mail Plan:", ["Yes", "No"])
        number_vmail_messages = st.number_input("Number of Voicemail Messages:", min_value=0, step=1)
        total_day_minutes = st.number_input("Total Day Minutes:", min_value=0.0, step=0.1)
        total_day_calls = st.number_input("Total Day Calls:", min_value=0, step=1)
        total_day_charge = st.number_input("Total Day Charge:", min_value=0.0, step=0.1)
        total_eve_minutes = st.number_input("Total Evening Minutes:", min_value=0.0, step=0.1)
        total_eve_calls = st.number_input("Total Evening Calls:", min_value=0, step=1)
        total_eve_charge = st.number_input("Total Evening Charge:", min_value=0.0, step=0.1)
        total_night_minutes = st.number_input("Total Night Minutes:", min_value=0.0, step=0.1)
        total_night_calls = st.number_input("Total Night Calls:", min_value=0, step=1)
        total_night_charge = st.number_input("Total Night Charge:", min_value=0.0, step=0.1)
        total_intl_minutes = st.number_input("Total International Minutes:", min_value=0.0, step=0.1)
        total_intl_calls = st.number_input("Total International Calls:", min_value=0, step=1)
        total_intl_charge = st.number_input("Total International Charge:", min_value=0.0, step=0.01)
        customer_service_calls = st.number_input("Customer Service Calls:", min_value=0, step=1)

        # Submit button
        submitted = st.form_submit_button("Predict")

        if submitted:
            # Prepare input data for prediction
            try:
                input_data = pd.DataFrame([{
                    "State": state,
                    "Account length": account_length,
                    "Area code": area_code,
                    "International plan": international_plan,
                    "Voice mail plan": voice_mail_plan,
                    "Number vmail messages": number_vmail_messages,
                    "Total day minutes": total_day_minutes,
                    "Total day calls": total_day_calls,
                    "Total day charge": total_day_charge,
                    "Total eve minutes": total_eve_minutes,
                    "Total eve calls": total_eve_calls,
                    "Total eve charge": total_eve_charge,
                    "Total night minutes": total_night_minutes,
                    "Total night calls": total_night_calls,
                    "Total night charge": total_night_charge,
                    "Total intl minutes": total_intl_minutes,
                    "Total intl calls": total_intl_calls,
                    "Total intl charge": total_intl_charge,
                    "Customer service calls": customer_service_calls
                }])

                # Get prediction
                prediction = model_pipeline.predict(input_data)
                probability = model_pipeline.predict_proba(input_data)

                # Display prediction
                st.success(f"Prediction: {'Churn' if prediction[0] else 'No Churn'}")
                st.info(f"Probability of Churn: {probability[0][1]:.2f}")
                st.info(f"Probability of No Churn: {probability[0][0]:.2f}")

            except Exception as e:
                st.error(f"An error occurred during prediction: {str(e)}")
                st.warning("Please ensure all inputs are valid and complete.")

if __name__ == "__main__":
    main()
