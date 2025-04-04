import streamlit as st
import pandas as pd
import numpy as np
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

st.markdown("Entrez les infos d’un passager pour voir s’il aurait survécu.")

sex = st.selectbox("Sexe", options=["Homme", "Femme"])
pclass = st.selectbox("Classe", options=[1, 2, 3])
age = st.slider("Âge", 0, 100, 25)

# Préparation des données pour prédiction
sex_encoded = 0 if sex == "Homme" else 1
input_data = np.array([[sex_encoded, pclass, age]])

# 3. Prédiction
if st.button("Prédire la survie"):
    prediction = model.predict(input_data)[0]
    probas = model.predict_proba(input_data)[0]
    
    if prediction == 1:
        st.success(f"✅ Ce passager aurait survécu (probabilité : {round(probas[1]*100, 2)}%)")
    else:
        st.error(f"❌ Ce passager n’aurait pas survécu (probabilité : {round(probas[0]*100, 2)}%)")
