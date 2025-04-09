# 🏡 Prédiction du prix de l'immobilier en Californie

Cette application Streamlit permet de prédire le **prix médian d'un bien immobilier** en Californie à partir de données sociodémographiques.  
Elle utilise un modèle de **Random Forest** entraîné sur le dataset `California Housing` (sklearn.datasets).

## 🚀 Fonctionnalités

- Interface simple et intuitive (en français 🇫🇷)
- Prédiction immédiate du prix en dollars
- Résumé des données saisies
- Explication des variables utilisées
- Déploiement via Streamlit Cloud

## 📁 Fichiers inclus

- `app.py` → application Streamlit
- `random_forest_model.pkl` → modèle entraîné
- `requirements.txt` → dépendances
- `README.md` → ce fichier

## ⚙️ Comment lancer l'app

1. Clonez ce dépôt
2. Lancez l'application avec :

```bash
streamlit run app.py
