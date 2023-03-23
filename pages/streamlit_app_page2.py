import streamlit as st
from index import load_data, load_model
import pandas as pd
from sklearn.inspection import partial_dependence
import plotly.express as px

st.title("Analyse du modèle entraîné")

df = load_data()
rf_model = load_model('model_rf_2011.pkl')

df_filter = df[
    (df['dteday'] <= '2011-12-31')
]
df_filter = df_filter[rf_model.feature_names_in_]

feature_importance_df = pd.DataFrame({'feature': rf_model.feature_names_in_,
                                      'importance': rf_model.feature_importances_})
feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

st.subheader("Contribution des variables explicatives")
st.dataframe(feature_importance_df)

st.subheader("Analyse de l'influence des variables sur les prédictions")

var = st.selectbox('Sélectionnez la variable explicative à analyser :', feature_importance_df['feature'])

pd_values = partial_dependence(rf_model, df_filter, [var], kind='average')
pd_df = pd.DataFrame({'average_prediction': pd_values['average'][0], var: pd_values['values'][0]})

fig = px.line(pd_df, x=var, y="average_prediction")
st.plotly_chart(fig)
