import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LogisticRegression

st.set_page_config(
    page_title="Titanic Survival App",  # ← le nom que tu veux voir dans l’onglet
    page_icon="🚢",                     # ← emoji ou URL d'icône perso
    layout="centered"
)

# 1. Entraînement du modèle
df = pd.read_csv("train.csv")

# Prétraitement léger
df = df[['Sex', 'Pclass', 'Age', 'Survived']].dropna()
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

X = df[['Sex', 'Pclass', 'Age']]
y = df['Survived']

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# 2. Interface utilisateur
st.title("🛳️ Titanic - Prédiction de survie")

st.subheader("🎯 Objectif")
st.markdown("Tester si un passager aurait survécu au naufrage du Titanic à partir de son profil.")

col1, col2, col3 = st.columns(3)

with col1:
    sex = st.selectbox("Sexe", options=["Homme", "Femme"])

with col2:
    pclass = st.selectbox("Classe", options=[1, 2, 3])

with col3:
    age = st.slider("Âge", 0, 100, 25)


# Préparation des données pour prédiction
sex_encoded = 0 if sex == "Homme" else 1
input_data = np.array([[sex_encoded, pclass, age]])

# 3. Prédiction
if st.button("Prédire la survie"):
    prediction = model.predict(input_data)[0]
    probas = model.predict_proba(input_data)[0]
    
    if prediction == 1:
        st.success(f"🎉 Ce passager aurait survécu !")
        st.markdown(f"🧠 Probabilité de survie : **{round(probas[1]*100, 2)}%**")
    else:
        st.error(f"💀 Ce passager n’aurait **pas** survécu...")
        st.markdown(f"📉 Probabilité de survie : **{round(probas[0]*100, 2)}%**")

st.subheader("📊 Répartition interactive des âges")

# Filtres utilisateur
col1, col2 = st.columns(2)

with col1:
    sexe_filtre = st.selectbox("Sexe", options=["Tous", "Homme", "Femme"])

with col2:
    classe_filtre = st.selectbox("Classe", options=["Toutes", 1, 2, 3])

# Copie et filtrage du dataframe
df_filtré = df.copy()

if sexe_filtre != "Tous":
    sexe_code = 0 if sexe_filtre == "Homme" else 1
    df_filtré = df_filtré[df_filtré['Sex'] == sexe_code]

if classe_filtre != "Toutes":
    df_filtré = df_filtré[df_filtré['Pclass'] == classe_filtre]

# Création du graphe filtré
fig = px.histogram(
    df_filtré,
    x='Age',
    nbins=30,
    title='Distribution des âges filtrée',
    color_discrete_sequence=['#4C78A8'],
    opacity=0.75
)

fig.update_layout(
    title_font_size=18,
    title_x=0.5,
    xaxis_title='Âge',
    yaxis_title='Nombre de passagers',
    plot_bgcolor='#f9f9f9',
    paper_bgcolor='#f9f9f9',
    bargap=0.05
)

fig.update_traces(marker_line_width=1, marker_line_color="white")

# Affichage du graphe final
st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("<p style='text-align:center;'>Réalisé par Aurélie PERNELLE | Avril 2025</p>", unsafe_allow_html=True)
