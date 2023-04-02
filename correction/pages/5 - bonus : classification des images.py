import streamlit as st
import numpy as np
from keras.applications import vgg19, VGG19
from keras.utils import load_img, img_to_array

# liste des modèles disponibles dans Keras
# https://keras.io/api/applications/

# zone d'upload d'image
st.subheader("Classification d'image")
upload_img = st.file_uploader("Choix de l'image", type=['png', 'jpg'])

# chargement du modèle pré-entraîné
model = VGG19()

if upload_img is not None:
    # chargement de l'image (pour affichage)
    img_original = load_img(upload_img)
    # chargement de l'image (pour prédiction)
    img_resized = load_img(upload_img, target_size=(224, 224))
    # affichage de l'image taille réelle
    st.image(img_original)
    # transposition de l'image dans un array pour traitement
    img_array = img_to_array(img_resized)
    # on ajoute une dimension servant à indiquer la taille du batch (obligatoire pour ingestion)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    # les données sont pré-processées pour ingestion dans le modèle prédictif (ex : normalisées...)
    img_array_expanded_dims_processed = vgg19.preprocess_input(img_array_expanded_dims)
    # calcul des prédictions et association des libellés de prédiction associés
    predictions = model.predict(img_array_expanded_dims)
    results = vgg19.decode_predictions(predictions)

    # affichage des résultats
    st.subheader("Résultats de la prédiction")
    for i in range(0, 5):
        st.write('{} : {}%'.format(results[0][i][1], np.round(results[0][i][2]*100, 2)))


