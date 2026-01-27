import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration ---
sns.set_theme(style="whitegrid")
os.makedirs('EDA/plots', exist_ok=True)

# 1. Chargement des données (Pandas)
print("--- Chargement des données ---")
df = pd.read_csv('EDA/sales_data.csv')
print(df.head())

# 2. Manipulation des données (Pandas & NumPy)
print("\n--- Analyse Statistique (NumPy / Pandas) ---")
# Moyenne des revenus par catégorie
revenue_stats = df.groupby('Category')['Revenue'].agg(['mean', 'sum', 'count']).round(2)
print(revenue_stats)

# Calcul d'un Z-score pour détecter les prix atypiques (Calcul mathématique avec NumPy / Pandas)
df['Price_ZScore'] = (df['Price'] - df['Price'].mean()) / df['Price'].std()
outliers = df[np.abs(df['Price_ZScore']) > 2] # Exemple simple d'outliers
print(f"\nNombre de valeurs atypiques (Z-score > 2): {len(outliers)}")

# 3. Visualisation (Matplotlib & Seaborn)
print("\n--- Génération des graphiques ---")

# Histogramme des Revenus
plt.figure(figsize=(10, 6))
sns.histplot(df['Revenue'], kde=True, color='skyblue')
plt.title('Distribution des Revenus')
plt.savefig('EDA/plots/distribution_revenus.png')
plt.close()

# Scatter Plot: Prix vs Rating
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Price', y='Rating', hue='Category', alpha=0.6)
plt.title('Prix vs Rating par Catégorie')
plt.savefig('EDA/plots/prix_vs_rating.png')
plt.close()

# Heatmap des corrélations
plt.figure(figsize=(10, 8))
numeric_df = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matrice de Corrélation')
plt.savefig('EDA/plots/correlation_heatmap.png')
plt.close()

print("\nAnalyse terminée. Les graphiques sont disponibles dans 'EDA/plots/'.")
