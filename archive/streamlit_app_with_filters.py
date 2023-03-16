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

price_range = [min(df['price']), max(df['price'])]
price_slider = st.sidebar.slider('Prix des logements', value=(price_range[0], price_range[1]))

neighbourhood_group_options = st.sidebar.multiselect(
    'Groupes de quartiers',
    list(df['neighbourhood_group'].unique()),
    list(df['neighbourhood_group'].unique())
)

df_filter = df[
    (df['price'] >= price_slider[0]) &
    (df['price'] <= price_slider[1]) &
    (df['neighbourhood_group'].isin(neighbourhood_group_options))
]
if len(df_filter) > 1000:
    df_sample = df_filter.sample(n=1000)
else:
    df_sample = df_filter.copy()

st.subheader('Données brutes')
st.write(df_filter)

st.subheader('Distribution des prix')


fig = px.histogram(df_filter, x='price', histnorm='probability', nbins=20)
st.plotly_chart(fig)


st.subheader('Moyenne des prix selon les groupes de quartiers')
fig = px.histogram(df_filter, x='neighbourhood_group', y='price', histfunc='avg')
st.plotly_chart(fig)

st.subheader('Moyenne des prix selon le type de logement')
fig = px.histogram(df_filter, x='room_type', y='price', histfunc='avg')
st.plotly_chart(fig)

st.subheader('Cartographie des logements')
fig = px.scatter_mapbox(df_sample, lat='latitude', lon='longitude', labels='labels',
                        mapbox_style="carto-positron", hover_name="name", color='price')


st.plotly_chart(fig)








