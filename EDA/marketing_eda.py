import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration de l'esthétique
sns.set_theme(style="whitegrid")
os.makedirs('EDA/marketing/plots', exist_ok=True)

print("--- Chargement des données Marketing ---")
# On utilise le séparateur point-virgule d'après les tests précédents
df = pd.read_csv('BDD/marketing_campaign.csv', sep=';')

# 1. Nettoyage et Préparation
# Calcul de l'âge du client (en supposant l'année actuelle 2024 pour l'exercice)
df['Age'] = 2024 - df['Year_Birth']

# Calcul des dépenses totales
mnt_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
df['Total_Spending'] = df[mnt_cols].sum(axis=1)

# Calcul du nombre total d'enfants
df['Children'] = df['Kidhome'] + df['Teenhome']

# 2. Analyse de Segmentation (Objectif : Segments rentables)

# Segmentation par Éducation et Revenu
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='Education', y='Income', palette='Set2')
plt.title('Distribution des Revenus par Niveau d\'Éducation')
plt.ylim(0, 150000) # Limite pour éviter les outliers extrêmes
plt.savefig('EDA/marketing/plots/income_by_education.png')
plt.close()

# Relation Âge vs Dépenses Totales
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Age', y='Total_Spending', hue='Marital_Status', alpha=0.5)
plt.title('Âge vs Dépenses Totales par Statut Marital')
plt.savefig('EDA/marketing/plots/age_vs_spending.png')
plt.close()

# 3. Analyse des Campagnes (Interaction)
campaign_cols = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Response']
campaign_sums = df[campaign_cols].sum()

plt.figure(figsize=(10, 6))
campaign_sums.plot(kind='bar', color='salmon')
plt.title('Taux de Réponse aux Campagnes de Marketing')
plt.ylabel('Nombre de Réponses Positives')
plt.savefig('EDA/marketing/plots/campaign_responses.png')
plt.close()

# 4. Matrice de Corrélation
plt.figure(figsize=(15, 10))
corr_cols = ['Age', 'Income', 'Total_Spending', 'Children', 'Recency', 'NumWebPurchases', 'NumStorePurchases']
sns.heatmap(df[corr_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Corrélation entre les Variables Clés')
plt.savefig('EDA/marketing/plots/correlation_matrix.png')
plt.close()

# Sauvegarde des données propres pour le dashboard
df.to_csv('EDA/marketing/cleaned_marketing_data.csv', index=False)

print("\nEDA Terminée. Résultats sauvegardés dans 'EDA/marketing/'.")
print(f"Total des clients analysés : {len(df)}")
print(f"Dépense moyenne par client : {df['Total_Spending'].mean():.2f} €")
