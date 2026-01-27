import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Configuration
st.set_page_config(page_title="Marketing Analytics Dashboard", layout="wide")
sns.set_theme(style="whitegrid")

st.title("🎯 Marketing Campaign Insights Dashboard")
st.markdown("""
Ce dashboard présente l'analyse des segments clients et l'efficacité des campagnes marketing.
Outils : **Pandas**, **NumPy**, **Seaborn**, **Plotly**.
""")

# Chargement
@st.cache_data
def load_data():
    if not os.path.exists('EDA/marketing/cleaned_marketing_data.csv'):
        return None
    return pd.read_csv('EDA/marketing/cleaned_marketing_data.csv')

import os
df = load_data()

if df is None:
    st.error("Veuillez d'abord exécuter 'py EDA/marketing_eda.py' pour générer les données nettoyées.")
else:
    # Sidebar Filters
    st.sidebar.header("Filtres Clients")
    education_filter = st.sidebar.multiselect("Niveau d'Éducation", options=df['Education'].unique(), default=df['Education'].unique())
    marital_filter = st.sidebar.multiselect("Statut Marital", options=df['Marital_Status'].unique(), default=df['Marital_Status'].unique())
    
    # Filtrage
    mask = df['Education'].isin(education_filter) & df['Marital_Status'].isin(marital_filter)
    filtered_df = df[mask]

    # KPIs
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Clients Sélectionnés", len(filtered_df))
    with kpi2:
        st.metric("Revenu Moyen", f"{filtered_df['Income'].mean():,.0f} €")
    with kpi3:
        st.metric("Dépense Totale", f"{filtered_df['Total_Spending'].sum():,.0f} €")
    with kpi4:
        st.metric("Taux d'Achat Web", f"{(filtered_df['NumWebPurchases'].sum() / filtered_df['NumStorePurchases'].sum() * 100):.1f}%")

    st.divider()

    # Visualisations
    row1_col1, row1_col2 = st.columns(2)

    with row1_col1:
        st.subheader("Rentabilité par Segment (Éducation)")
        fig_rent = px.box(filtered_df, x="Education", y="Total_Spending", color="Education", notched=True)
        st.plotly_chart(fig_rent, use_container_width=True)

    with row1_col2:
        st.subheader("Âge vs Dépenses (Interactif)")
        fig_age = px.scatter(filtered_df, x="Age", y="Total_Spending", size="Income", color="Marital_Status", hover_name="ID", opacity=0.6)
        st.plotly_chart(fig_age, use_container_width=True)

    st.divider()

    row2_col1, row2_col2 = st.columns([1, 2])

    with row2_col1:
        st.subheader("Efficacité des Campagnes")
        campaign_cols = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Response']
        camp_data = filtered_df[campaign_cols].sum().reset_index()
        camp_data.columns = ['Campagne', 'Acceptations']
        fig_camp = px.bar(camp_data, x='Campagne', y='Acceptations', color='Acceptations', scale_both_axis=True)
        st.plotly_chart(fig_camp, use_container_width=True)

    with row2_col2:
        st.subheader("Répartition des Dépenses par Produit")
        mnt_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
        product_spending = filtered_df[mnt_cols].mean().reset_index()
        product_spending.columns = ['Produit', 'Moyenne Dépensée']
        fig_prod = px.pie(product_spending, values='Moyenne Dépensée', names='Produit', hole=.3)
        st.plotly_chart(fig_prod, use_container_width=True)

    st.divider()
    
    if st.checkbox("Afficher les données brutes"):
        st.dataframe(filtered_df)
