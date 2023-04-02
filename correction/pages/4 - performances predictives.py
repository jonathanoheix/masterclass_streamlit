import streamlit as st
from index import load_data, load_model
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

st.title("Analyse des performances prédictives")

# chargement des données
df = load_data()

selected_model = st.selectbox('Modèle sélectionné', ('model_rf_2011', 'model_tree_2011'))

# chargement du modèle
model = load_model('{}.joblib'.format(selected_model))

df_predict = df[df['dteday'].dt.year == 2012].copy()

# calcul des prédictions et de l'erreur absolue avec les valeurs réelles
df_predict['prediction'] = model.predict(df_predict[model.feature_names_in_])
df_predict['absolute_error'] = np.abs(df_predict['prediction'] - df_predict['cnt'])

# nuage de points
st.subheader("Comparaison des valeurs réelles et prédites")
fig = px.scatter(df_predict, x="cnt", y="prediction", color="absolute_error", color_continuous_scale="thermal")
st.plotly_chart(fig)

# métriques d'évaluation
st.subheader("Métriques d'évaluation des prédictions")
col1, col2, col3, col4 = st.columns(4)
col1.metric('RMSE', np.round(np.sqrt(mean_squared_error(df_predict['prediction'], df_predict['cnt'])), 2))
col2.metric('MAE', np.round(mean_absolute_error(df_predict['prediction'], df_predict['cnt']), 2))
col3.metric('MAPE', '{} %'.format(
    np.round(mean_absolute_percentage_error(df_predict['prediction'], df_predict['cnt']) * 100, 2)))
col4.metric('ME', np.round(np.mean(df_predict['prediction'] - df_predict['cnt']), 2))

# comparaison des valeurs des deux années
st.subheader("Evolution des valeurs de la variable cible")
df_cnt = df[['dteday', 'cnt']].copy()
df_cnt['year'] = df_cnt['dteday'].dt.year.astype(str)
df_cnt['date_str'] = df_cnt['dteday'].dt.strftime("%m/%d")
fig = px.scatter(df_cnt, x="date_str", y="cnt", color="year")
st.plotly_chart(fig)
