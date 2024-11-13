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

# Load models and encoders
@st.cache_resource
def load_models_and_encoders():
    models = {
        'Random Forest': joblib.load('rf_best_gb_model.pkl'),
        'Gradient Boost': joblib.load('gb_best_gb_model.pkl'),
        'Decision Tree': joblib.load('dt_best_gb_model.pkl')
    }
    # Load separate label encoders for each model
    encoders = {
        'Random Forest': pickle.load(open('rf_label_encoders.pkl', 'rb')),
        'Gradient Boost': pickle.load(open('gb_label_encoders.pkl', 'rb')),
        'Decision Tree': pickle.load(open('dt_label_encoders.pkl', 'rb'))
    }
    return models, encoders

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
    # Load models and encoders
    models, label_encoders = load_models_and_encoders()
    
    # Main title
    st.title('🦌 CWD Prediction System')
    st.markdown('---')

    # Model Selection
    st.markdown('<div class="model-selection">', unsafe_allow_html=True)
    st.subheader('Select Prediction Model')
    selected_model = st.radio(
        "Choose a model for prediction:",
        ['Random Forest', 'Gradient Boost', 'Decision Tree'],
        help="Each model has different characteristics and may produce slightly different results"
    )
    
    # Display model information
    model_info = {
        'Random Forest': "Ensemble learning method that combines multiple decision trees",
        'Gradient Boost': "Sequential ensemble method that builds upon weak learners",
        'Decision Tree': "Single tree model with straightforward decision rules"
    }
    st.info(f"Selected Model: {model_info[selected_model]}")
    st.markdown('</div>', unsafe_allow_html=True)

    # Get selected model and its corresponding encoders
    model = models[selected_model]
    current_encoders = label_encoders[selected_model]
    
    # Get original feature columns from the selected model
    original_columns = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else None
    
    if original_columns is None:
        st.error("Could not determine original feature columns from the model")
        st.stop()

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

        # Preprocess input and make prediction
        try:
            processed_input = preprocess_input(input_data, current_encoders, original_columns)
            
            # Make prediction using selected model
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

        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.info("Please check if all input values are in the correct format and try again.")

except Exception as e:
    st.error(f"An error occurred while loading the models: {str(e)}")
    st.warning("Please ensure all model files are properly loaded and input data is correct.")

# Footer
st.markdown('---')
st.markdown("""
<div style='text-align: center'>
    <p>© 2024 Virginia Department of Wildlife Resources</p>
</div>
""", unsafe_allow_html=True)