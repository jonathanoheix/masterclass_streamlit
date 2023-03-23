import streamlit as st
from index import load_data, load_model
import pandas as pd
import numpy as np

st.title("Démo du modèle")

df = load_data()
rf_model = load_model('model_rf_2011.pkl')

if 'last_prediction' not in st.session_state:
    st.session_state['last_prediction'] = None

temp_val = st.slider('Valeur de température', min(df['temp']), max(df['temp']), float(df['temp'].mean()))
hum_val = st.slider("Valeur d'humidité", min(df['hum']), max(df['hum']), float(df['hum'].mean()))
windspeed_val = st.slider('Valeur de vitesse du vent', min(df['windspeed']), max(df['windspeed']),
                          float(np.mean(df['windspeed'])))
weekday_val = st.selectbox('Valeur du jour de la semaine :', df['weekday'].unique())

parameters = {}

for f in rf_model.feature_names_in_:
    if f == 'temp':
        parameters[f] = temp_val
    elif f == 'hum':
        parameters[f] = hum_val
    elif f == 'windspeed':
        parameters[f] = windspeed_val
    elif f == 'weekday_{}'.format(weekday_val):
        parameters[f] = 1
    else:
        parameters[f] = 0

pred_value_df = pd.DataFrame(parameters, index=[0])
prediction = rf_model.predict(pred_value_df)[0]

if st.session_state['last_prediction'] is None:
    delta = None
else:
    delta = np.round(prediction - st.session_state['last_prediction'], 2)

st.metric('Prédiction du modèle', prediction, delta=delta)
st.session_state['last_prediction'] = prediction
