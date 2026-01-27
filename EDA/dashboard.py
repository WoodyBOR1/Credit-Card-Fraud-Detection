import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Configuration de la page
st.set_page_config(page_title="Dashboard EDA Ventes", layout="wide")

st.title("📊 Dashboard d'Analyse Exploratoire des Ventes")
st.markdown("""
Ce tableau de bord interactif présente les résultats de l'EDA effectués sur les données de ventes.
Il utilise **Pandas**, **NumPy**, **Matplotlib** et **Seaborn**.
""")

# Chargement des données
@st.cache_data
def load_data():
    df = pd.read_csv('EDA/sales_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()

# Barre latérale (Filtres)
st.sidebar.header("Filtres")
selected_category = st.sidebar.multiselect("Sélectionnez les catégories", options=df['Category'].unique(), default=df['Category'].unique())
selected_location = st.sidebar.multiselect("Sélectionnez les villes", options=df['Location'].unique(), default=df['Location'].unique())

# Filtrage du dataframe
filtered_df = df[(df['Category'].isin(selected_category)) & (df['Location'].isin(selected_location))]

# Section 1: KPIs
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Revenu", f"{filtered_df['Revenue'].sum():,.2f} €")
with col2:
    st.metric("Nombre de Ventes", f"{len(filtered_df)}")
with col3:
    st.metric("Prix Moyen", f"{filtered_df['Price'].mean():.2f} €")
with col4:
    st.metric("Note Moyenne", f"{filtered_df['Rating'].mean():.2f} ⭐")

st.divider()

# Section 2: Visualisations
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Distribution des Ventes par Catégorie")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=filtered_df, x='Category', palette='viridis', ax=ax1)
    plt.xticks(rotation=45)
    st.pyplot(fig1)

with col_b:
    st.subheader("Revenu Cumulé dans le Temps")
    # Groupement par date
    revenue_time = filtered_df.groupby('Date')['Revenue'].sum().reset_index().sort_values('Date')
    fig2, ax2 = plt.subplots()
    sns.lineplot(data=revenue_time, x='Date', y='Revenue', color='orange', ax=ax2)
    plt.xticks(rotation=45)
    st.pyplot(fig2)

st.divider()

# Section 3: Analyse de Corrélation
st.subheader("Analyse des Variables")
option = st.selectbox("Afficher la corrélation pour :", ['Globale', 'Prix vs Rating', 'Tableau de données'])

if option == 'Globale':
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    numeric_df = filtered_df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), annot=True, cmap='RdBu', ax=ax3)
    st.pyplot(fig3)
elif option == 'Prix vs Rating':
    fig4 = sns.jointplot(data=filtered_df, x='Price', y='Rating', kind="hex", color="#4CB391")
    st.pyplot(fig4.fig)
else:
    st.dataframe(filtered_df.head(50))

st.sidebar.info("Dashboard créé pour le projet Data Science.")
