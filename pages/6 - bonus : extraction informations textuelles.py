from spacy_streamlit import visualize_ner
import spacy
import streamlit as st

st.subheader("Reconnaissance d'entités nommées")

# chargement du modèle d'analyse NLP Spacy
nlp = spacy.load("fr_core_news_lg")

# zone de saisie
txt = st.text_area("Saisir le texte à analyser", "Apple est créée le 1er avril 1976 dans le garage de la maison d'enfance de Steve Jobs à Los Altos en Californie par Steve Jobs, Steve Wozniak et Ronald Wayne, puis constituée sous forme de société le 3 janvier 1977 à l'origine sous le nom d'Apple Computer. Cependant, pour ses 30 ans et pour refléter la diversification de ses produits, le mot « computer » est retiré le 9 janvier 2007.",
                   height=200)

# initialisation du traitement textuel
doc = nlp(txt)
# mise en place de la visualisation avec Streamlit
visualize_ner(doc, labels=nlp.get_pipe("ner").labels)

