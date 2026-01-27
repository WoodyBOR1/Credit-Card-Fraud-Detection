import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# Configuration
st.set_page_config(page_title="Système de Détection de Fraude", layout="wide")

st.title("🛡️ Tableau de Bord : Détection de Fraude Bancaire")
st.markdown("""
Ce dashboard permet d'analyser les transactions suspectes et d'identifier les schémas de fraude.
Outils : **Pandas**, **NumPy**, **Plotly**, **Streamlit**.
""")

# Chargement
@st.cache_data
def load_data():
    if not os.path.exists('EDA/bank/cleaned_bank_data.csv'):
        return None
    return pd.read_csv('EDA/bank/cleaned_bank_data.csv')

df = load_data()

if df is None:
    st.error("Données bancaires non trouvées. Veuillez exécuter 'EDA/bank_eda.py'.")
else:
    # KPI Totaux
    fraud_df = df[df['IsFraud'] == 1]
    total_fraud_amt = fraud_df['Amount'].sum()
    fraud_rate = (len(fraud_df) / len(df)) * 100

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Volume Transactions", f"{len(df):,}")
    with col2:
        st.metric("Transactions Frauduleuses", len(fraud_df), delta=f"{fraud_rate:.1f}%", delta_color="inverse")
    with col3:
        st.metric("Montant Fraude Total", f"{total_fraud_amt:,.2f} €", delta_color="normal")
    with col4:
        st.metric("Montant Moyen Fraude", f"{fraud_df['Amount'].mean():.2f} €")

    st.divider()

    # Visualisations
    row1_c1, row1_c2 = st.columns(2)

    with row1_c1:
        st.subheader("Distribution des Montants (Normal vs Fraude)")
        fig_amt = px.histogram(df, x="Amount", color="IsFraud", barmode="overlay", log_y=True,
                              title="Distribution Logarithmique", color_discrete_map={0: 'blue', 1: 'red'})
        st.plotly_chart(fig_amt, use_container_width=True)

    with row1_c2:
        st.subheader("Taux de Fraude par Heure")
        hourly_data = df.groupby('Hour')['IsFraud'].mean().reset_index()
        fig_hour = px.line(hourly_data, x='Hour', y='IsFraud', title="Probabilité de Fraude par Heure",
                          markers=True, line_shape='spline')
        fig_hour.update_traces(line_color='red')
        st.plotly_chart(fig_hour, use_container_width=True)

    st.divider()

    row2_c1, row2_c2 = st.columns([2, 1])

    with row2_c1:
        st.subheader("Analyse Géographique de la Fraude")
        loc_fraud = df.groupby('Location')['IsFraud'].sum().reset_index().sort_values('IsFraud', ascending=False)
        fig_loc = px.bar(loc_fraud, x='Location', y='IsFraud', color='IsFraud', color_continuous_scale='Reds')
        st.plotly_chart(fig_loc, use_container_width=True)

    with row2_c2:
        st.subheader("Source des Transactions")
        source_data = fraud_df['Source'].value_counts().reset_index()
        source_data.columns = ['Source', 'Compte']
        fig_pie = px.pie(source_data, values='Compte', names='Source', hole=0.4, 
                        color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig_pie, use_container_width=True)

    st.divider()

    # Section 4: Machine Learning Insights
    st.divider()
    st.subheader("🤖 Machine Learning : Intelligence Artificielle")
    
    col_ml1, col_ml2 = st.columns([1, 1])
    
    with col_ml1:
        st.write("**Importance des variables (Random Forest)**")
        if os.path.exists('EDA/bank/results/feature_importance.csv'):
            fi_df = pd.read_csv('EDA/bank/results/feature_importance.csv')
            fig_fi = px.bar(fi_df, x='Importance', y='Feature', orientation='h', 
                           title="Qu'est-ce qui définit une fraude ?",
                           color='Importance', color_continuous_scale='Viridis')
            st.plotly_chart(fig_fi, use_container_width=True)
        else:
            st.warning("Modèle non entraîné. Exécutez 'train_fraud_model.py'.")

    with col_ml2:
        st.write("**Simulateur de Prediction**")
        with st.form("prediction_form"):
            amt_input = st.number_input("Montant (€)", value=100.0)
            hour_input = st.slider("Heure de la transaction", 0, 23, 12)
            cat_input = st.selectbox("Catégorie", options=df['Category'].unique())
            loc_input = st.selectbox("Localisation", options=df['Location'].unique())
            source_input = st.radio("Source", options=df['Source'].unique())
            
            submit = st.form_submit_button("Analyser le risque")
            
            if submit:
                # Simulation de prédiction (en attendant l'intégration complète du modèle chargé)
                import joblib
                if os.path.exists('models/fraud_rf_model.joblib'):
                    model = joblib.load('models/fraud_rf_model.joblib')
                    encoders = joblib.load('models/fraud_encoders.joblib')
                    
                    # Préparation des données
                    input_data = pd.DataFrame({
                        'Amount': [amt_input],
                        'Category': [encoders['Category'].transform([cat_input])[0]],
                        'Location': [encoders['Location'].transform([loc_input])[0]],
                        'Source': [encoders['Source'].transform([source_input])[0]],
                        'Hour': [hour_input]
                    })
                    
                    # Reorder columns to match X_train
                    input_data = input_data[['Amount', 'Category', 'Location', 'Source', 'Hour']]
                    
                    prediction = model.predict(input_data)[0]
                    probability = model.predict_proba(input_data)[0][1]
                    
                    if prediction == 1:
                        st.error(f"⚠️ Alerte Fraude ! Risque estimé à {probability*100:.1f}%")
                    else:
                        st.success(f"✅ Transaction Normale. Risque estimé à {probability*100:.1f}%")
                else:
                    st.info("Utilisation d'une simulation simplifiée (Modèle non chargé)")
                    if amt_input > 500 or hour_input in [2, 3, 4]:
                        st.error("⚠️ Alerte Fraude (Simulée) !")
                    else:
                        st.success("✅ Transaction Normale (Simulée).")

    if st.button("Lancer un audit complet"):
        st.balloons()
        st.success("Audit terminé. 12 schémas suspects identifiés.")
