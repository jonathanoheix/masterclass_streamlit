import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Analyse de l'utilisation de vélo en libre service à Washington DC (2011)")


@st.cache_data
def load_data():
    data = pd.read_csv('Bike-Sharing-Dataset/day.csv')
    data = data[['dteday', 'holiday', 'weekday', 'weathersit', 'temp', 'hum', 'windspeed', 'cnt', 'season']]
    data['dteday'] = pd.to_datetime(data['dteday'], format='%Y-%m-%d')
    data['temp'] = data['temp'] * 41
    data['hum'] = data['hum'] * 100
    data['windspeed'] = data['windspeed'] * 67
    data['weekday'] = data['dteday'].dt.day_name()
    data['season'] = data['season'].map({1: 'winter', 2: 'spring', 3: 'summer', 4: 'fall'})
    return data


df = load_data()
df_filter = df[df['dteday'] <= '2011-12-31'].copy()
df_filter['month'] = df_filter['dteday'].dt.month_name()
df_filter['weekday'] = df_filter['dteday'].dt.day_name()

st.subheader("Evolution du nombre d'utilisations quotidiennes")
fig = px.scatter(df_filter, x="dteday", y="cnt", color="temp", color_continuous_scale="thermal")
st.plotly_chart(fig)

fig = px.violin(df_filter, y="cnt", x="season", color="season", box=True,
                color_discrete_sequence=["#87CEEB", "#90EE90", "#FFA07A", "#FFDAB9"])
st.plotly_chart(fig)

fig = px.histogram(df_filter, x='weekday', y='cnt', histfunc='avg')
st.plotly_chart(fig)

fig = px.scatter(df_filter, x="windspeed", y="cnt", trendline="ols")
st.plotly_chart(fig)





