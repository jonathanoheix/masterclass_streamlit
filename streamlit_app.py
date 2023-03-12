import streamlit as st
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.title('Analyse des prix des Airbnbs à New York (2019)')


@st.cache_data
def load_data():
    data = pd.read_csv('AB_NYC_2019.csv')
    return data


df = load_data()
df_sample = df.sample(frac=0.01)

st.subheader('Données brutes')
st.write(df)

st.subheader('Distribution des prix')


fig = px.histogram(df, x='price', histnorm='probability', nbins=20)
st.plotly_chart(fig)


st.subheader('Moyenne des prix selon les groupes de quartiers')
fig = px.histogram(df, x='neighbourhood_group', y='price', histfunc='avg')
st.plotly_chart(fig)

st.subheader('Moyenne des prix selon le type de logement')
fig = px.histogram(df, x='room_type', y='price', histfunc='avg')
st.plotly_chart(fig)

st.subheader('Cartographie des logements')
fig = px.scatter_mapbox(df_sample, lat='latitude', lon='longitude', labels='labels',
                        mapbox_style="carto-positron", hover_name="name")


st.plotly_chart(fig)








