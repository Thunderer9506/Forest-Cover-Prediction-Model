# main.py
import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score

# --- Page Configuration (should be the first st command) ---
st.set_page_config(
    page_title="Vehicle Price Predictor",
    page_icon="🔮",
    layout='wide'
)

# --- Caching the Model and Preprocessing Objects ---
@st.cache_data
def setup_and_train_model():
    """
    Performs all data loading, cleaning, feature engineering, encoding, scaling,
    and model training. Caches the final objects needed for prediction.
    """
    # 1. LOAD DATA
    data1 = pd.read_csv('train.csv',index_col="Id") # Got from internship
    data2 = pd.read_csv('covtype.csv') # got from internet
    forestData = pd.concat([data2,data1],ignore_index=True)

    
    # 4. MODEL TRAINING
    X = forestData.drop(['Cover_Type'],axis=1)
    y = forestData['Cover_Type']
    

    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    model = rf_classifier.fit(X, y)
    
    return model

# --- Load all artifacts from the setup function ---
model = setup_and_train_model()

# --- Prediction Function ---
def predict_price(input_data):
    """
    Takes a dictionary of user inputs, applies the full preprocessing pipeline,
    and returns a price prediction.
    """
    df = pd.DataFrame([input_data])
    
    # Scale and predict
    prediction = model.predict(df)
    
    return round(float(prediction[0]), 2)

# --- Streamlit User Interface ---
st.title("🌴 Forest Cover Type Prediction Model")
st.sidebar.success("1. Select your features\n2. Click 'Predict'")
st.sidebar.divider()
st.sidebar.header("Your Selections:")

col1, col2 = st.columns([1, 1.5])

with col2:
    st.header("Select the Features")
    
    # # --- Dynamic and Dependent Dropdowns ---
    elevation = st.slider('Elevation',1860,3860,2596)
    aspect = st.slider('Aspect',0,360,51)
    slope = st.slider('Slope',0,66,3)
    HDHydrolysis = st.slider('Horizontal_Distance_To_Hydrology',0,1400,258)
    VDHydrolysis = st.slider('Vertical_Distance_To_Hydrology',-170,600,0)
    HDRoadways = st.slider('Horizontal_Distance_To_Roadways',0,7110,510)
    HDFirePoints = st.slider('Horizontal_Distance_To_Fire_Points',0,7170,6279)
    Hillshade_9am = st.slider('Hillshade_9am',0,250,221)
    Hillshade_3pm = st.slider('Hillshade_3pm',0,250,148)
    Hillshade_Noon = st.slider('Hillshade_Noon',0,250,232)
    Wilderness_Area = st.selectbox('Wilderness Area',[f"Wilderness_Area{i}" for i in range(1,5)],0)
    Soil_Type = st.selectbox('Soil Type',[f"Soil_Type{i}" for i in range(1,41)],28)

# Display user choices in the sidebar
selections = {
    "Elevation": elevation,
    "Aspect":aspect ,
    "Slope":slope,
    "Horizontal_Distance_to_Hydrolysis": HDHydrolysis,
    "Vertical_Distance_to_Hydrolysis": VDHydrolysis,
    "Horizontal_Distance_to_Roadways": HDRoadways,
    "Horizontal_Distance_to_FirePoints": HDFirePoints,
    "Hillshade_9am": Hillshade_9am,
    "Hillshade_3pm": Hillshade_3pm,
    "Hillshade_Noon": Hillshade_Noon,
    "Wilderness_Area": Wilderness_Area,
    "Soil_Type": Soil_Type,
}

for label, value in selections.items():
    st.sidebar.write(f"**{label}:** {value}")

with col1:
    st.header("Predicted Cover Type")
    forestType = ['Spruce/Fir','Lodgepole Pine','Ponderosa Pine','Aspen','Douglas-fir','Krummholz']
    if st.button("Predict 🔮", use_container_width=True):
        input_sliderbar = {
            "Elevation": elevation,
            "Aspect":aspect ,
            "Slope":slope,
            "Horizontal_Distance_To_Hydrology": HDHydrolysis,
            "Vertical_Distance_To_Hydrology": VDHydrolysis,
            "Horizontal_Distance_To_Roadways": HDRoadways,
            "Hillshade_9am": Hillshade_9am,
            "Hillshade_Noon": Hillshade_Noon,
            "Hillshade_3pm": Hillshade_3pm,
            "Horizontal_Distance_To_Fire_Points": HDFirePoints,
        }
        input_wilderness = {f"Wilderness_Area{i}":0 for i in range(1,5)}
        input_soilType = {f"Soil_Type{i}":0 for i in range(1,41)}
        input_wilderness[Wilderness_Area] = 1
        input_soilType[Soil_Type] = 1
        input_data = input_sliderbar | input_wilderness | input_soilType
        cType = predict_price(input_data)
        
        st.metric("Cover Type", f"Cover Type {forestType[int(cType)-1]}")
    else:
        st.metric("Cover Type", forestType[0])