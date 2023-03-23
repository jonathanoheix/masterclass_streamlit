import streamlit as st
import pandas as pd
from joblib import  load

st.title("Bienvenue sur la webapp ! 👋")


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
    data['month'] = data['dteday'].dt.month_name()
    data['weekday'] = data['dteday'].dt.day_name()
    data = data.join(pd.get_dummies(data['weekday'], prefix='weekday'))
    return data


def load_model(path):
    model = load(path)
    return model

