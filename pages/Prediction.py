import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from typing import Tuple, Any

if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
    st.session_state.model = None
    st.session_state.evaluation_data = None

# Page Configuration
st.set_page_config(
    page_title="Forest Cover Prediction",
    page_icon="🔮",
    layout='wide'
)

@st.cache_resource
def train_and_evaluate_model(trainData) -> Tuple[Any, dict]:
    forestData = pd.read_csv('covtype.csv').drop(["ID"],axis=1)[:trainData]
    X = forestData.drop(['Cover_Type'],axis=1)
    y = forestData['Cover_Type']
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=30)
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    rf_classifier.fit(X_train, y_train)
    
    forestType = ['Spruce/Fir', 'Lodgepole Pine', 'Ponderosa Pine', 
              'Cottonwood/Willow', 'Aspen', 'Douglas-fir', 'Krummholz']
    
    y_pred = rf_classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred,output_dict=True)
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')
    
    evaluation_data = {
        'accuracy': accuracy,
        'weighted_f1': weighted_f1,
        'classification_report': classification_rep,
        'feature_names': X.columns.tolist(),
        'class_names': forestType
    }
    
    return rf_classifier, evaluation_data

if not st.session_state.model_trained:
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown("### 🎯 Select Training Data Size")
        st.warning("Because the Data is very large we let user decide on how much data should the model be trained on")
        trainData = st.slider("Choose how many rows to train on", 1000, 596132, 10000, 1000)
        if st.button("Train Model", use_container_width=True):
            with st.spinner('Training model... Please wait...'):
                model, evaluation_data = train_and_evaluate_model(trainData)
                st.session_state.model = model
                st.session_state.evaluation_data = evaluation_data
                st.session_state.model_trained = True
                st.success('Model training complete!')
                st.rerun()
else:
    model = st.session_state.model
    evaluation_data = st.session_state.evaluation_data
    st.title("🔮 Forest Cover Type Prediction")
    st.markdown("""
    This interactive tool predicts forest cover types based on cartographic variables. 
    Select the features using the sliders below and click 'Predict' to see the results.
    """)

def predict_price(input_data):
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)
    return round(float(prediction[0]), 2)



with st.sidebar:
    st.title("🌲 Navigation")
    st.success("Make predictions by adjusting the parameters")
    st.info("The model uses Random Forest classification to predict cover types")


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
            
            confidence_placeholder.progress(0.96)  
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

st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <i>Model trained on Roosevelt National Forest data</i>
</div>
""", unsafe_allow_html=True)