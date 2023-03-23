import streamlit as st
from index import load_data, load_model
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

st.title("Analyse des performances prédictives")

df = load_data()
rf_model = load_model('model_rf_2011.pkl')

year = st.sidebar.slider('Année de prédiction', min(df['dteday'].dt.year), max(df['dteday'].dt.year),
                         max(df['dteday'].dt.year))
temp_range = st.sidebar.slider('Valeurs de température', min(df['temp']), max(df['temp']),
                               [min(df['temp']), max(df['temp'])])
hum_range = st.sidebar.slider("Valeurs d'humidité", min(df['hum']), max(df['hum']), [min(df['hum']), max(df['hum'])])
windspeed_range = st.sidebar.slider('Valeurs de vitesse du vent', min(df['windspeed']), max(df['windspeed']),
                                    [min(df['windspeed']), max(df['windspeed'])])
weekday_values = st.sidebar.multiselect('Jours de la semaine', df['weekday'].unique(), df['weekday'].unique())

df = df[
    (df['temp'].between(temp_range[0], temp_range[1])) &
    (df['hum'].between(hum_range[0], hum_range[1])) &
    (df['windspeed'].between(windspeed_range[0], windspeed_range[1])) &
    (df['weekday'].isin(weekday_values))
]

df_predict = df[df['dteday'].dt.year == year].copy()

df_predict['prediction'] = rf_model.predict(df_predict[rf_model.feature_names_in_])
df_predict['absolute_error'] = np.abs(df_predict['prediction'] - df_predict['cnt'])

st.subheader("Comparaison des valeurs réelles et prédites")
fig = px.scatter(df_predict, x="cnt", y="prediction", color="absolute_error", color_continuous_scale="thermal")
st.plotly_chart(fig)

st.subheader("Métriques d'évaluation des prédictions")
col1, col2, col3, col4 = st.columns(4)
col1.metric('RMSE', np.round(np.sqrt(mean_squared_error(df_predict['prediction'], df_predict['cnt'])), 2))
col2.metric('MAE', np.round(mean_absolute_error(df_predict['prediction'], df_predict['cnt']), 2))
col3.metric('MAPE', '{} %'.format(
    np.round(mean_absolute_percentage_error(df_predict['prediction'], df_predict['cnt'])*100, 2)))
col4.metric('ME', np.round(np.mean(df_predict['prediction'] - df_predict['cnt']), 2))


st.subheader("Evolution des valeurs de la variable cible")
df_cnt = df[['dteday', 'cnt']].copy()
df_cnt['year'] = df_cnt['dteday'].dt.year.astype(str)
df_cnt['date_str'] = df_cnt['dteday'].dt.strftime("%m/%d")
fig = px.scatter(df_cnt, x="date_str", y="cnt", color="year")
st.plotly_chart(fig)


