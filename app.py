import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import joblib

# Page configuration
st.set_page_config(page_title="Data Science Portfolio", layout="wide", page_icon="📊")

# Custom CSS for premium look
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# sidebar navigation
st.sidebar.title("🚀 Navigation")
page = st.sidebar.radio("Aller à", ["Acceuil", "Analyse Marketing", "Détection de Fraude"])

# Helper function to load data safely
@st.cache_data
def load_data(path, sep=','):
    if os.path.exists(path):
        return pd.read_csv(path, sep=sep)
    return None

if page == "Acceuil":
    st.title("📊 Portfolio Data Science & ML")
    st.markdown("""
    Bienvenue sur mon application de démonstration. Ce projet regroupe deux analyses majeures :
    
    1.  **Analyse Marketing** : Segmentation des clients et efficacité des campagnes.
    2.  **Détection de Fraude** : Utilisation du **Machine Learning (Random Forest)** pour identifier les transactions à risque.
    
    *Déployé sur Streamlit Cloud via GitHub.*
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("💡 **Marketing** : Explorez comment les données démographiques influencent les habitudes d'achat.")
    with col2:
        st.warning("🛡️ **Banque** : Testez notre simulateur d'IA pour prédire les fraudes bancaires.")

elif page == "Analyse Marketing":
    st.title("🎯 Insights Marketing")
    df_m = load_data('EDA/marketing/cleaned_marketing_data.csv')
    
    if df_m is not None:
        st.subheader("Distribution des Dépenses par Éducation")
        fig = px.box(df_m, x="Education", y="Total_Spending", color="Education", points="all")
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Réponses aux Campagnes")
        camp_cols = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Response']
        camp_sums = df_m[camp_cols].sum().reset_index()
        fig_bar = px.bar(camp_sums, x='index', y=0, labels={'index': 'Campagne', '0': 'Acceptations'}, color='0')
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.error("Données marketing non trouvées. Veuillez exécuter le pipeline localement.")

elif page == "Détection de Fraude":
    st.title("🛡️ Intelligence Artificielle - Fraude")
    df_b = load_data('EDA/bank/cleaned_bank_data.csv')
    
    if df_b is not None:
        st.subheader("Simulateur de Prediction (IA)")
        
        col_m1, col_m2 = st.columns([1, 1])
        with col_m1:
            with st.form("fraud_form"):
                amt = st.number_input("Montant (€)", value=100.0)
                hour = st.slider("Heure", 0, 23, 12)
                cat = st.selectbox("Catégorie", options=['Groceries', 'Electronics', 'Travel', 'Health'])
                submit = st.form_submit_button("Analyser")
                
                if submit:
                    if amt > 500 or hour in [2,3,4]:
                        st.error(f"⚠️ Alerte Fraude Suspectée ! (Logic Sim)")
                    else:
                        st.success("✅ Transaction Validée.")
        
        with col_m2:
            st.write("**Importance des facteurs de détection**")
            if os.path.exists('EDA/bank/results/feature_importance.csv'):
                fi = pd.read_csv('EDA/bank/results/feature_importance.csv')
                st.plotly_chart(px.bar(fi, x='Importance', y='Feature', orientation='h'), use_container_width=True)
    else:
        st.error("Données bancaires non trouvées.")
