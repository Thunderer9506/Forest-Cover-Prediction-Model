import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io

st.set_page_config(
    page_title="Home",
    page_icon="👋",
    layout='centered'
)

st.sidebar.success("🌴 Forest Cover Type Prediction Model")

data1 = pd.read_csv('train.csv',index_col="Id") # Got from internship
data2 = pd.read_csv('covtype.csv') # got from internet
forestData = pd.concat([data2,data1],ignore_index=True)

st.title("Forest Cover Type Prediction Model")

st.write("## Statistics")

st.write('### Shape of the data: ')
st.write(f'Rows : {forestData.shape[0]}')
st.write(f'Cols( {len(list(forestData.columns))} ) : `{", ".join(list(forestData.columns))}`')

buffer = io.StringIO()
forestData.info(buf=buffer)
info = buffer.getvalue()

st.divider()

st.write('### Data Info')
st.text(info[37:])

st.write('### Data Head')
st.write(forestData.head())

st.divider()

st.write('### Data Described')
st.write(forestData.describe())

st.write('## Exploratory Data Analysis')

st.write('### Forest Cover Type Distibution')

fig = plt.figure()
ax = sns.countplot(data=forestData,x='Cover_Type',order=[2,1,3,7,6,5,4])

ax.set_xlabel('Cover_Type')
ax.set_ylabel('Count')
ax.set_ylim(0,300000)
ax.set_xticks([0,1,2,3,4,5,6],[0,1,2,3,4,5,6])
plt.tight_layout()
for i,j in enumerate(forestData['Cover_Type'].value_counts()):
    ax.text(i, j, str(j),
            fontsize = 8,
            ha='center',
            va='bottom',
            )

st.pyplot(fig)

st.divider()

st.write('### Feature Distributions')

fig = forestData[['Elevation','Aspect','Slope','Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology',
           'Horizontal_Distance_To_Roadways','Horizontal_Distance_To_Fire_Points','Hillshade_9am','Hillshade_Noon',
           'Hillshade_3pm']].hist(bins=50, figsize=(20,15), layout=(4,3))
plt.tight_layout()
st.pyplot(fig[0][0].figure)

st.divider()

st.write('### Geospatial Relationships')
st.text('🌍 Since this is geographical data, features like Horizontal_Distance_To_Roadways, Vertical_Distance_To_Hydrology, etc., may relate spatially.')

fig = plt.figure()
sns.scatterplot(x='Horizontal_Distance_To_Hydrology', y='Vertical_Distance_To_Hydrology', hue='Cover_Type', data=forestData)
fig.tight_layout()
st.pyplot(fig)

st.divider()

st.write('### Wilderness Area vs Cover Type analysis')

fig,axes = plt.subplots(2,2,figsize = (16,10))
ax1 = sns.countplot(x='Wilderness_Area1', hue='Cover_Type', data=forestData,ax=axes[0,0])
ax2 = sns.countplot(x='Wilderness_Area2', hue='Cover_Type', data=forestData,ax=axes[0,1])
ax3 = sns.countplot(x='Wilderness_Area3', hue='Cover_Type', data=forestData,ax=axes[1,0])
ax4 = sns.countplot(x='Wilderness_Area4', hue='Cover_Type', data=forestData,ax=axes[1,1])
ax1.set_title('Wilderness1 vs Cover_type',fontdict={'size':'18','weight':'600'})
ax2.set_title('Wilderness2 vs Cover_type',fontdict={'size':'18','weight':'600'})
ax3.set_title('Wilderness3 vs Cover_type',fontdict={'size':'18','weight':'600'})
ax4.set_title('Wilderness4 vs Cover_type',fontdict={'size':'18','weight':'600'})
plt.tight_layout()

st.pyplot(fig)

st.divider()

st.write('### Soil Type vs Cover type relation')

soil_cols = [f"Soil_Type{i}" for i in range(1,41)]
soil_onehot = forestData[soil_cols]

# get soil type label (e.g. 'Soil_Type7') then convert to integer 7
soil_type_series = soil_onehot.idxmax(axis=1).str.replace('Soil_Type', '').astype(int)

fig, ax = plt.subplots()
pd.crosstab(soil_type_series, forestData['Cover_Type']).plot(kind='bar', stacked=True, ax=ax)
ax.set_xlabel('Soil_Type')
ax.set_ylabel('Count')
ax.set_title('Soil Type vs Cover Type', fontdict={'size':'18','weight':'600'})
plt.tight_layout()

st.pyplot(fig)

st.divider()

st.write("## Results")
st.text('1. Forest Contain Large Number of Spruce/Fir Trees')
st.text('2. Trees are usually 3000m Long')
st.text('3. Trees are generally near to river/lake')
st.text('4. Trees get more Hillshade during 9 am and Noon')
st.text('5. Tree 2 can grow in any type of soil specially in type 29')