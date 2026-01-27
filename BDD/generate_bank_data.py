import pandas as pd
import numpy as np
import os

# Create BDD directory if it doesn't exist
os.makedirs('BDD', exist_ok=True)

# Set seed for reproducibility
np.random.seed(42)

# Generate synthetic Bank Transaction data
n_rows = 5000
transaction_ids = [f"TX{i:05d}" for i in range(n_rows)]
customer_ids = [f"CUST{np.random.randint(100, 500):03d}" for _ in range(n_rows)]
dates = pd.date_range(start='2024-01-01', periods=180, freq='D')
transaction_dates = np.random.choice(dates, n_rows)

categories = ['Groceries', 'Electronics', 'Travel', 'Entertainment', 'Health', 'Restaurant']
locations = ['Paris', 'Lyon', 'Marseille', 'London', 'New York', 'Tokyo', 'Berlin']

# Base Data
df = pd.DataFrame({
    'TransactionID': transaction_ids,
    'CustomerID': customer_ids,
    'Date': transaction_dates,
    'Amount': np.round(np.random.exponential(scale=50, size=n_rows) + 5, 2),
    'Category': np.random.choice(categories, n_rows),
    'Location': np.random.choice(locations, n_rows),
    'Source': np.random.choice(['Online', 'POS', 'ATM'], n_rows, p=[0.4, 0.5, 0.1]),
    'IsFraud': 0
})

# Inject Artificial Fraud Patterns
# Pattern 1: High amount transactions
fraud_indices = df.sample(int(n_rows * 0.02)).index
df.loc[fraud_indices, 'IsFraud'] = 1
df.loc[fraud_indices, 'Amount'] = df.loc[fraud_indices, 'Amount'] * np.random.uniform(5, 10)

# Pattern 2: International transactions for specific customers
strange_customers = df['CustomerID'].unique()[:5]
intl_fraud = df[df['CustomerID'].isin(strange_customers)].sample(frac=0.3).index
df.loc[intl_fraud, 'IsFraud'] = 1
df.loc[intl_fraud, 'Location'] = 'Unknown/Tax Haven'

# Pattern 3: Late night transactions (Artificial Hour)
df['Hour'] = np.random.randint(0, 24, n_rows)
late_night = df[df['Hour'].isin([2, 3, 4])].sample(frac=0.1).index
df.loc[late_night, 'IsFraud'] = 1

# Save to CSV
df.to_csv('BDD/bank_transactions.csv', index=False)
print("Dataset 'BDD/bank_transactions.csv' generated successfully for Fraud Analysis.")
