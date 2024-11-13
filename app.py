import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="CWD Prediction System",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {padding: 20px;}
    .stButton>button {width: 100%; margin-top: 20px;}
    </style>
    """, unsafe_allow_html=True)

# Load model and encoders
@st.cache_resource
def load_model_and_encoders():
    model = joblib.load('rf_best_gb_model.pkl')
    with open('rf_label_encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    return model, encoders

def preprocess_input(data, label_encoders, original_columns):
    processed_data = data.copy()
    
    # Handle categorical variables with unseen labels
    for column, le in label_encoders.items():
        if column in processed_data.columns:
            processed_data[column] = processed_data[column].apply(
                lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else -1
            )
    
    # Ensure all features are present and in the correct order
    processed_data = processed_data[original_columns]
    
    return processed_data

try:
    # Load model and encoders
    model, label_encoders = load_model_and_encoders()
    
    # Get original feature columns from the model
    original_columns = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else None
    
    if original_columns is None:
        st.error("Could not determine original feature columns from the model")
        st.stop()
    
    # Main title
    st.title('🦌 CWD Prediction System')
    st.markdown('---')

    # Create two columns for input
    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Sample Information')
        deer_id = st.text_input('Deer ID', 'B14441')
        sample_date = st.date_input('Sample Date', datetime.now())
        sample_date_str = f"{sample_date} 00:00:00.000"
        
        sample_type = st.selectbox(
            'Sample Type',
            ['Active - Hunter Killed', 'Targeted - Other', 'Targeted-Clinical suspect', 'Active - Road-Killed']
        )
        sex = st.selectbox('Sex', ['Male', 'Female', 'Unknown'])

    with col2:
        st.subheader('Location and Age Details')
        age = st.number_input('Age (years)', min_value=0.0, max_value=20.0, value=1.5, step=0.5)
        mile_grid = st.number_input('1 Mile Grid', min_value=0, max_value=40000, value=32440)
        fips_county = st.text_input('FIPS County', 'Franklin County')

    # Create prediction button
    if st.button('Predict CWD Status'):
        # Create input DataFrame
        input_data = pd.DataFrame({
            'DeerID': [deer_id],
            'SampleDate': [sample_date_str],
            'Sample': [sample_type],
            'Sex': [sex],
            'Age': [age],
            '1MileGrid': [mile_grid],
            'FIPSCounty': [fips_county]
        })

        # Preprocess input
        try:
            processed_input = preprocess_input(input_data, label_encoders, original_columns)
            
            # Make prediction
            prediction = model.predict(processed_input)
            probability = model.predict_proba(processed_input)

            # Display results
            st.markdown('---')
            st.subheader('Prediction Results')

            # Create columns for results
            res_col1, res_col2 = st.columns(2)

            with res_col1:
                if 'Detected' in prediction[0]:
                    st.error(f'🚨 Prediction: {prediction[0]}')
                else:
                    st.success(f'✅ Prediction: {prediction[0]}')

            with res_col2:
                # Display probabilities
                st.write('Probability Distribution:')
                prob_df = pd.DataFrame({
                    'Result': model.classes_,
                    'Probability': probability[0]
                })
                st.dataframe(prob_df.style.format({'Probability': '{:.2%}'}))

            # Display warning for unseen labels if any were encountered
            if (processed_input == -1).any().any():
                st.warning("""
                ⚠️ Some input values were not seen during model training. 
                These have been handled, but predictions may be less reliable.
                """)

        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.info("Please check if all input values are in the correct format and try again.")

except Exception as e:
    st.error(f"An error occurred while loading the model: {str(e)}")
    st.warning("Please ensure all model files are properly loaded and input data is correct.")

# Footer
st.markdown('---')
st.markdown("""
<div style='text-align: center'>
    <p>Developed for CWD Surveillance and Monitoring</p>
    <p>© 2024 Virginia Department of Wildlife Resources</p>
</div>
""", unsafe_allow_html=True)