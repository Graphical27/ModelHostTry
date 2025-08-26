# streamlit_app.py
import json
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import streamlit as st
from datetime import datetime

DEFAULT_API_BASE = 'http://localhost:8000'
CSV_PATH = 'crop_yield.csv'

if 'api_base' not in st.session_state:
    st.session_state.api_base = DEFAULT_API_BASE

@st.cache_data
def load_dataset_unique_values(csv_path: str):
    if not Path(csv_path).exists():
        return {
            'Crop': [], 'Season': [], 'State': [], 'Crop_Year': []
        }
    try:
        df = pd.read_csv(csv_path)
        values = {
            'Crop': sorted([str(x) for x in df['Crop'].dropna().unique()]) if 'Crop' in df.columns else [],
            'Season': sorted([str(x) for x in df['Season'].dropna().unique()]) if 'Season' in df.columns else [],
            'State': sorted([str(x) for x in df['State'].dropna().unique()]) if 'State' in df.columns else [],
            'Crop_Year': sorted([int(x) for x in df['Crop_Year'].dropna().unique()]) if 'Crop_Year' in df.columns else [],
        }
        return values
    except Exception as e:
        st.warning(f"Could not load dataset options: {e}")
        return {'Crop': [], 'Season': [], 'State': [], 'Crop_Year': []}

def call_api_predict(api_base: str, payload: dict):
    predict_url = f"{api_base}/predict"
    try:
        response = requests.post(predict_url, json=payload, timeout=30)
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"API error {response.status_code}: {response.text}"
    except requests.exceptions.Timeout:
        return None, "Request timed out. Please check if the API is running."
    except requests.exceptions.ConnectionError:
        return None, f"Could not connect to API at {api_base}. Is the FastAPI server running?"
    except Exception as e:
        return None, f"Request failed: {str(e)}"

def call_api_health(api_base: str):
    health_url = f"{api_base}/health"
    try:
        response = requests.get(health_url, timeout=10)
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"Health check failed {response.status_code}: {response.text}"
    except requests.exceptions.ConnectionError:
        return None, f"Could not connect to API at {api_base}"
    except Exception as e:
        return None, f"Health check failed: {str(e)}"

st.set_page_config(
    page_title='Crop Yield Predictor',
    page_icon='🌾',
    layout='centered',
    initial_sidebar_state='expanded'
)

st.title('🌾 Crop Yield Prediction System')
st.markdown('### Predict agricultural yield using machine learning')
st.caption('Enter crop details below to get yield predictions with confidence intervals')

with st.sidebar:
    st.markdown('### 🔧 API Configuration')

    api_base = st.text_input(
        'FastAPI Base URL',
        value=st.session_state.api_base,
        help='URL where your FastAPI server is running'
    )
    st.session_state.api_base = api_base

    if st.button('🔍 Check API Health', use_container_width=True):
        with st.spinner('Checking API health...'):
            health_data, error = call_api_health(api_base)
            if health_data:
                st.success('✅ API is healthy!')
                st.json(health_data)
            else:
                st.error(f'❌ API health check failed: {error}')
    
    st.markdown('---')
    st.markdown('### 📋 Instructions')
    st.markdown("""
    1. **Start FastAPI server**: Run `uvicorn app:app --reload`
    2. **Fill in crop details** in the form below
    3. **Click Predict** to get yield estimates
    4. View prediction with confidence intervals
    """)

unique_vals = load_dataset_unique_values(CSV_PATH)

st.markdown('### 📝 Crop Information')

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('**Basic Information**')

        crop_options = unique_vals.get('Crop', [])
        if crop_options:
            crop_choice = st.selectbox('🌱 Crop Type', options=[''] + crop_options)
            crop_custom = st.text_input('Or enter custom crop', placeholder='e.g., Wheat, Rice, Corn')
            crop_val = crop_custom.strip() if crop_custom.strip() else crop_choice
        else:
            crop_val = st.text_input('🌱 Crop Type *', placeholder='e.g., Wheat, Rice, Corn')

        year_options = unique_vals.get('Crop_Year', [])
        if year_options:
            current_year = datetime.now().year
            default_year_idx = 0
            if current_year in year_options:
                default_year_idx = year_options.index(current_year)
            elif year_options:

                default_year_idx = min(range(len(year_options)), 
                                     key=lambda i: abs(year_options[i] - current_year))
            
            year_choice = st.selectbox('📅 Crop Year', options=[None] + year_options, 
                                     index=default_year_idx + 1 if year_options else 0)
        else:
            year_choice = st.number_input('📅 Crop Year', min_value=1900, max_value=2030, 
                                        value=datetime.now().year, step=1)

        season_options = unique_vals.get('Season', [])
        season_choice = st.selectbox('🌦️ Season', options=[None] + season_options) if season_options else st.text_input('🌦️ Season', placeholder='e.g., Kharif, Rabi')
        
        state_options = unique_vals.get('State', [])
        state_choice = st.selectbox('🗺️ State', options=[None] + state_options) if state_options else st.text_input('🗺️ State', placeholder='e.g., Punjab, Uttar Pradesh')
    
    with col2:
        st.markdown('**Agricultural Parameters**')

        area = st.number_input('📏 Area (hectares) *', min_value=0.01, value=1.0, step=0.1, 
                              help='Cultivated area in hectares')

        st.markdown('**Optional Parameters**')
        rainfall = st.number_input('🌧️ Annual Rainfall (mm)', min_value=0.0, value=0.0, step=10.0,
                                 help='Total annual rainfall in millimeters')
        
        fertilizer = st.number_input('🧪 Fertilizer (kg/ha)', min_value=0.0, value=0.0, step=1.0,
                                   help='Fertilizer usage per hectare')
        
        pesticide = st.number_input('🚫 Pesticide (kg/ha)', min_value=0.0, value=0.0, step=0.1,
                                  help='Pesticide usage per hectare')

    st.markdown('---')
    submitted = st.form_submit_button('🎯 Predict Crop Yield', use_container_width=True, type='primary')

if submitted:

    if not crop_val or not crop_val.strip():
        st.error('❌ Please specify a crop type')
        st.stop()
    
    if area <= 0:
        st.error('❌ Area must be greater than 0')
        st.stop()

    payload = {
        'Crop': crop_val.strip(),
        'Crop_Year': int(year_choice) if year_choice else None,
        'Season': season_choice if season_choice else None,
        'State': state_choice if state_choice else None,
        'Area': float(area),
        'Annual_Rainfall': float(rainfall) if rainfall > 0 else None,
        'Fertilizer': float(fertilizer) if fertilizer > 0 else None,
        'Pesticide': float(pesticide) if pesticide > 0 else None,
    }

    with st.spinner('🔄 Making prediction...'):
        prediction_data, error = call_api_predict(st.session_state.api_base, payload)
    
    if prediction_data:
        st.markdown('### 📊 Prediction Results')

        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label='Predicted Yield',
                value=f"{prediction_data.get('prediction', 0):,.2f} kg/ha"
            )
        
        with col2:
            st.metric(
                label='Lower Bound (95%)',
                value=f"{prediction_data.get('lower_95', 0):,.2f} kg/ha"
            )
        
        with col3:
            st.metric(
                label='Upper Bound (95%)',
                value=f"{prediction_data.get('upper_95', 0):,.2f} kg/ha"
            )

        st.markdown('#### 📈 Confidence Interval')
        prediction = prediction_data.get('prediction', 0)
        lower = prediction_data.get('lower_95', 0)
        upper = prediction_data.get('upper_95', 0)
        
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 2))
        
        ax.barh([0], [upper - lower], left=lower, height=0.3, alpha=0.3, color='lightblue', label='95% Confidence Interval')
        ax.plot([prediction], [0], 'ro', markersize=10, label=f'Prediction: {prediction:,.2f}')
        
        ax.set_yticks([])
        ax.set_xlabel('Yield (kg/ha)')
        ax.set_title('Crop Yield Prediction with Confidence Interval')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        with st.expander('📋 Request Details', expanded=False):
            st.markdown('**Payload sent to API:**')
            st.json(payload)
        
        with st.expander('🔍 Full API Response', expanded=False):
            st.json(prediction_data)
        
        if prediction_data.get('model'):
            st.info(f"🤖 Model used: {prediction_data['model']}")
    
    else:
        st.error(f'❌ Prediction failed: {error}')
        st.markdown('**Troubleshooting:**')
        st.markdown('1. Ensure FastAPI server is running on the specified URL')
        st.markdown('2. Check that the model file exists and is properly loaded')
        st.markdown('3. Verify the input data format matches model expectations')


st.markdown('---')
st.markdown(
    '<div style="text-align: center; color: gray;">'
    'Crop Yield Prediction System | Built with Streamlit + FastAPI'
    '</div>',
    unsafe_allow_html=True
)