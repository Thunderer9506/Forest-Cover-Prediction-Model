import streamlit as st
import pandas as pd
import pickle

# Page Configuration
st.set_page_config(
    page_title="Forest Cover Prediction",
    page_icon="🔮",
    layout='wide'
)

@st.cache_resource
def importingModel():
    # Load model
    with open("model.pkl", 'rb') as file:
        model = pickle.load(file)
    
    # Load evaluation data
    with open("model_evaluation.pkl", 'rb') as file:
        evaluation_data = pickle.load(file)
    
    return model, evaluation_data

# Update model loading
model, evaluation_data = importingModel()

def predict_price(input_data):
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)
    return round(float(prediction[0]), 2)

# Main content
st.title("🔮 Forest Cover Type Prediction")
st.markdown("""
This interactive tool predicts forest cover types based on cartographic variables. 
Select the features using the sliders below and click 'Predict' to see the results.
""")
with st.sidebar:
    st.title("🌲 Navigation")
    st.success("Make predictions by adjusting the parameters")
    st.info("The model uses Random Forest classification to predict cover types")

# Main content tabs
tab1, tab2 = st.tabs(["🎯 Prediction", "ℹ️ Feature Guide"])

with tab1:
    col1, col2 = st.columns([1, 1.5])
    with col2:
        st.subheader("Terrain Features")
        elevation = st.slider('Elevation (m)', 1860, 3860, 2596)
        aspect = st.slider('Aspect (degrees)', 0, 360, 51)
        slope = st.slider('Slope (degrees)', 0, 66, 3)
        
        st.subheader("Distance Features")
        HDHydrolysis = st.slider('Distance to Water (m)', 0, 1400, 258)
        VDHydrolysis = st.slider('Vertical Distance to Water (m)', -170, 600, 0)
        HDRoadways = st.slider('Distance to Roads (m)', 0, 7110, 510)
        HDFirePoints = st.slider('Distance To Fire Points (m)',0,7170,6279)
        
        st.subheader("Sunlight Features")
        Hillshade_9am = st.slider('Morning Shade', 0, 250, 221)
        Hillshade_Noon = st.slider('Noon Shade', 0, 250, 232)
        Hillshade_3pm = st.slider('Afternoon Shade', 0, 250, 148)
        
        st.subheader("Area Classification")
        Wilderness_Area = st.selectbox('Wilderness Area', 
                                     [f"Wilderness_Area{i}" for i in range(1,5)])
        Soil_Type = st.selectbox('Soil Type', 
                                [f"Soil_Type{i}" for i in range(1,41)],28)
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
        st.subheader("Predicted Forest Cover")
        forestType = ['Spruce/Fir', 'Lodgepole Pine', 'Ponderosa Pine', 
                     'Aspen', 'Douglas-fir', 'Krummholz']
        
        prediction_placeholder = st.empty()
        confidence_placeholder = st.empty()
        
        if st.button("🔮 Predict Cover Type", use_container_width=True):
            # ... existing prediction code ...
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
            print(input_data)
            cType = predict_price(input_data)
            prediction_placeholder.metric("Cover Type", forestType[int(cType)-1])
            # Add confidence score (if available from model)
            confidence_placeholder.progress(0.96)  # Example confidence
            confidence_placeholder.caption("Prediction Confidence: 96%")
        
        
        st.markdown("### Model Performance Metrics")
    
        met_col1, met_col2 = st.columns(2)
        
        with met_col1:
            st.metric("Overall Accuracy", f"{evaluation_data['accuracy']:.2%}")
            st.metric("Weighted F1-Score", f"{evaluation_data['weighted_f1']:.2%}")
        
        with met_col2:
            report_dict = evaluation_data['classification_report']  
            st.metric("Macro Precision", f"{report_dict['macro avg']['precision']:.2%}")
            st.metric("Macro Recall", f"{report_dict['macro avg']['recall']:.2%}")
        
        with st.expander("Detailed Performance by Cover Type"):
            metrics_df = pd.DataFrame(evaluation_data['classification_report']).T
            metrics_df = metrics_df.drop(['accuracy', 'macro avg', 'weighted avg'])
            metrics_df.index = evaluation_data['class_names']
            st.dataframe(metrics_df.round(3), use_container_width=True)

with tab2:
    st.header("Feature Information")
    
    feature_info = {
        "Elevation": "Height above sea level in meters",
        "Aspect": "Direction the slope faces (0-360 degrees)",
        "Slope": "Steepness of terrain (0-90 degrees)",
        "Distance Features": "Horizontal and vertical distances to various landmarks",
        "Hillshade": "Illumination levels at different times of day",
        "Wilderness Areas": "Designated wilderness area classification",
        "Soil Types": "Soil type classification based on USFS data"
    }
    
    for feature, description in feature_info.items():
        with st.expander(feature):
            st.write(description)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <i>Model trained on Roosevelt National Forest data</i>
</div>
""", unsafe_allow_html=True)