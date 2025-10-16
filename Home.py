import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import io

# Page Configuration
st.set_page_config(
    page_title="Forest Cover Analysis Dashboard",
    page_icon="🌲",
    layout='wide'  # Changed to wide layout for better visualization
)


@st.cache_data
def importData():
    return pd.read_csv('covtype.csv').drop(["ID"],axis=1)

forestData = importData()

# Add this function with your other cached functions
@st.cache_data
def get_dataset_stats(df):
    stats = {
        "Total Records": f"{df.shape[0]:,}",
        "Total Features": f"{len(df.columns)}",
        "Numerical Features": f"{len(df.select_dtypes(include=['int64', 'float64']).columns)}",
        "Categorical Features": f"{len(df.select_dtypes(include=['object', 'bool']).columns)}",
        "Missing Values": f"{df.isnull().sum().sum():,}",
        "Memory Usage": f"{df.memory_usage().sum() / 1024**2:.2f} MB",
        "Unique Cover Types": f"{df['Cover_Type'].nunique()}",
        "Most Common Cover Type": f"Type {df['Cover_Type'].mode()[0]} ({df['Cover_Type'].value_counts().iloc[0]:,} samples)",
        "Elevation Range": f"{df['Elevation'].min():,} - {df['Elevation'].max():,} meters",
    }
    return stats

@st.cache_data
def covertypePlot():
    fig = px.histogram(forestData,x="Cover_Type",title="Cover Type")
    fig.update_layout(
        bargap = 0.2,
        xaxis=dict(
            tickmode='linear', 
            tick0=1,
        )
    )
    return fig

@st.cache_data
def featuredistributionPlots():
    st.write("#### Terrain Features")
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))
    forestData[['Elevation','Aspect','Slope',
                'Hillshade_9am','Hillshade_Noon','Hillshade_3pm']].hist(bins=50,ax=axes)
    plt.tight_layout()
    return fig


@st.cache_data
def distancerelatedFeatures():
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 10))
    forestData[['Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology',
                'Horizontal_Distance_To_Roadways','Horizontal_Distance_To_Fire_Points']].hist(bins=50,ax=axes)
    plt.tight_layout()
    return fig

@st.cache_data
def hydorlogyScatterPlot():
    fig = plt.figure()
    sns.scatterplot(x='Horizontal_Distance_To_Hydrology', y='Vertical_Distance_To_Hydrology', 
                    hue='Cover_Type', data=forestData)
    return fig

@st.cache_data
def wildernessPlot():
    fig,axes = plt.subplots(2,2,figsize = (16,12))
    ax1 = sns.countplot(x='Wilderness_Area1', hue='Cover_Type', data=forestData,ax=axes[0,0])
    ax2 = sns.countplot(x='Wilderness_Area2', hue='Cover_Type', data=forestData,ax=axes[0,1])
    ax3 = sns.countplot(x='Wilderness_Area3', hue='Cover_Type', data=forestData,ax=axes[1,0])
    ax4 = sns.countplot(x='Wilderness_Area4', hue='Cover_Type', data=forestData,ax=axes[1,1])
    ax1.set_title('Wilderness1 vs Cover_type',fontdict={'size':'18','weight':'600'})
    ax2.set_title('Wilderness2 vs Cover_type',fontdict={'size':'18','weight':'600'})
    ax3.set_title('Wilderness3 vs Cover_type',fontdict={'size':'18','weight':'600'})
    ax4.set_title('Wilderness4 vs Cover_type',fontdict={'size':'18','weight':'600'})
    plt.tight_layout()
    return fig

@st.cache_data
def soilCoverType():
    soil_cols = [f"Soil_Type{i}" for i in range(1,41)]
    soil_onehot = forestData[soil_cols]

    # get soil type label (e.g. 'Soil_Type7') then convert to integer 7
    soil_type_series = soil_onehot.idxmax(axis=1).str.replace('Soil_Type', '').astype(int)
    
    # Create figure first
    fig = plt.figure(figsize=(12,6))
    
    # Create the crosstab and plot it
    pd.crosstab(soil_type_series, forestData['Cover_Type']).plot(
        kind='bar', 
        stacked=True,
        ax=fig.gca()  # get current axes
    )
    
    plt.xlabel('Soil_Type')
    plt.ylabel('Count')
    plt.title('Soil Type vs Cover Type', fontdict={'size':'18', 'weight':'600'})
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig

# Sidebar
with st.sidebar:
    st.title("🌳 Navigation")
    st.success("Forest Cover Type Analysis")
    st.info("This application analyzes forest cover types based on cartographic variables.")
    
    # Add download option for sample data
    st.download_button(
        label="Download Sample Data",
        data=forestData.sample(5).to_csv(),
        file_name="sample_forest_data.csv",
        mime="text/csv"
    )


# Main Content
st.title("🌲 Forest Cover Type Analysis Dashboard")
st.markdown("""
This dashboard provides comprehensive analysis of forest cover types based on various geographical 
and cartographic variables. The dataset contains observations of Roosevelt National Forest in Colorado.
""")

# Data Overview Tab
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "📈 Feature Analysis", "🗺️ Geospatial Insights", "🔍 Key Findings"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Dataset Statistics")
        stats = get_dataset_stats(forestData)
        
        for key, value in stats.items():
            st.markdown(f"""
                <b>{key}:</b> {value}
            """, unsafe_allow_html=True)
        
    with col2:
        st.write("### Class Distribution")
        # Your existing cover type distribution plot
        st.plotly_chart(covertypePlot())
        
    
    st.write("### Sample Data")
    st.dataframe(forestData.head(), use_container_width=True)

with tab2:
    st.write("### Feature Distributions")
    # Your existing feature distribution plots
    st.pyplot(featuredistributionPlots())

    st.write("#### Distance Features")
    # Show distance-related features
   
    st.pyplot(distancerelatedFeatures())

with tab3:
    st.write("### Geographic Relationships")
    
    # Your existing geospatial plot with improvements
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("#### Hydrology Analysis")
        # Your existing hydrology scatter plot
        
        st.pyplot(hydorlogyScatterPlot())
        
    with col2:
        st.write("#### Wilderness Areas")
        # Your existing wilderness area analysis
        # make it for other wilderness area too
        st.pyplot(wildernessPlot())

    st.write("#### Soil vs Cover Type")
    st.pyplot(soilCoverType())

with tab4:
    st.write("### Key Insights")
    
    # Convert your results into more engaging format
    insights = {
        "Dominant Species": "Spruce/Fir trees are predominant in the forest",
        "Elevation Pattern": "Most trees are found at around 3000m elevation",
        "Water Proximity": "Trees show clustering near water bodies",
        "Sunlight Exposure": "Maximum hillshade during morning and noon",
        "Soil Adaptability": "Cover Type 2 shows high adaptability across soil types"
    }
    
    for title, description in insights.items():
        with st.expander(title):
            st.write(description)

# Add footer
st.markdown("""---""")
st.markdown("""
<div style='text-align: center'>
    <i>Data Source: Roosevelt National Forest, Colorado</i>
</div>
""", unsafe_allow_html=True)