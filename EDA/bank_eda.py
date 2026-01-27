import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
sns.set_theme(style="darkgrid")
os.makedirs('EDA/bank/plots', exist_ok=True)

print("--- Chargement des données Bancaires ---")
df = pd.read_csv('BDD/bank_transactions.csv')

# 1. Analyse de la Distribution du Fraude
fraud_counts = df['IsFraud'].value_counts()
fraud_rate = (fraud_counts[1] / len(df)) * 100

print(f"Nombre de transactions : {len(df)}")
print(f"Taux de fraude : {fraud_rate:.2f}%")

# 2. Visualisation des Montants (Fraude vs Normal)
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='IsFraud', y='Amount', palette='Reds')
plt.title('Distribution des Montants : Transactions Normales vs Fraude')
plt.yscale('log') # Utilisation de l'échelle log pour voir les différences
plt.savefig('EDA/bank/plots/amount_distribution_fraud.png')
plt.close()

# 3. Analyse Temporelle (Par Heure)
plt.figure(figsize=(12, 6))
hourly_fraud = df.groupby(['Hour', 'IsFraud']).size().unstack().fillna(0)
hourly_fraud_rate = (hourly_fraud[1] / (hourly_fraud[0] + hourly_fraud[1])) * 100
hourly_fraud_rate.plot(kind='line', marker='o', color='red')
plt.title('Taux de Fraude par Heure de la Journée')
plt.xlabel('Heure')
plt.ylabel('Taux de Fraude (%)')
plt.savefig('EDA/bank/plots/fraud_by_hour.png')
plt.close()

# 4. Analyse Géographique (Top Locations Fraude)
plt.figure(figsize=(10, 6))
top_fraud_locations = df[df['IsFraud'] == 1]['Location'].value_counts().head(5)
top_fraud_locations.plot(kind='bar', color='darkred')
plt.title('Top 5 Localisations de Fraude')
plt.savefig('EDA/bank/plots/top_fraud_locations.png')
plt.close()

# 5. Corrélations
plt.figure(figsize=(10, 8))
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title('Matrice de Corrélation - Données Bancaires')
plt.savefig('EDA/bank/plots/bank_correlation.png')
plt.close()

# Sauvegarde des données propres pour le dashboard
df.to_csv('EDA/bank/cleaned_bank_data.csv', index=False)

print("\nEDA Bancaire terminée. Résultats sauvegardés dans 'EDA/bank/'.")
