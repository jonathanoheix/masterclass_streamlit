import datetime
import streamlit as st
import plotly.express as px
import numpy as np
from index import load_data

st.title("Analyse de l'utilisation de vélos en libre service à Washington DC (2011)")

# appel des données
df = load_data()

# champs de filtrage
temp_range = st.sidebar.slider('Valeurs de température', min(df['temp']), max(df['temp']),
                               [min(df['temp']), max(df['temp'])])
hum_range = st.sidebar.slider("Valeurs d'humidité", min(df['hum']), max(df['hum']), [min(df['hum']), max(df['hum'])])
windspeed_range = st.sidebar.slider('Valeurs de vitesse du vent', min(df['windspeed']), max(df['windspeed']),
                                    [min(df['windspeed']), max(df['windspeed'])])
weekday_values = st.sidebar.multiselect('Jours de la semaine', df['weekday'].unique(), df['weekday'].unique())
date_values = st.sidebar.date_input('Dates', [datetime.date(2011, 1, 1), datetime.date(2011, 12, 31)],
                                    datetime.date(2011, 1, 1), datetime.date(2011, 12, 31))

# si aucune date de fin n'est renseignée, on assigne le 31/12/2011
if len(date_values) == 1:
    date_values = (date_values[0], datetime.date(2011, 12, 31))

# filtrage des données
df_filter = df[
    (df['dteday'].between(date_values[0].strftime('%Y-%m-%d'), date_values[1].strftime('%Y-%m-%d'))) &
    (df['temp'].between(temp_range[0], temp_range[1])) &
    (df['hum'].between(hum_range[0], hum_range[1])) &
    (df['windspeed'].between(windspeed_range[0], windspeed_range[1])) &
    (df['weekday'].isin(weekday_values))
    ]

# affichage des données
st.subheader("Aperçu des données")
st.dataframe(df.head(100))

# nuage de points
st.subheader("Evolution du nombre d'utilisations quotidiennes")
fig = px.scatter(df_filter, x="dteday", y="cnt", color="temp", color_continuous_scale="thermal")
st.plotly_chart(fig)

# violin plot
st.subheader("Distribution du nombre d'utilisations selon la saison")
fig = px.violin(df_filter, y="cnt", x="season", color="season", box=True,
                color_discrete_sequence=["#87CEEB", "#90EE90", "#FFA07A", "#FFDAB9"])
st.plotly_chart(fig)

# diagramme en barres
st.subheader("Moyenne du nombre d'utilisations selon le jour de la semaine")
fig = px.histogram(df_filter, x='weekday', y='cnt', histfunc='avg')
st.plotly_chart(fig)

# nuage de points
st.subheader("Lien entre la vitesse du vent et le nombre d'utilisations")
fig = px.scatter(df_filter, x="windspeed", y="cnt", trendline="ols")
st.plotly_chart(fig)

# métrique
st.metric(label="Corrélation entre la vitesse du vent et le nombre d'utilisations",
          value=np.round(df_filter['cnt'].corr(df_filter['windspeed']), 2))
