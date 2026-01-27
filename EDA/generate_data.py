import pandas as pd
import numpy as np
import os

# Create EDA directory if it doesn't exist
os.makedirs('EDA', exist_ok=True)

# Set seed for reproducibility
np.random.seed(42)

# Generate synthetic Sales data
n_rows = 1000
dates = pd.date_range(start='2023-01-01', periods=n_rows, freq='D')
categories = ['Electronics', 'Clothing', 'Home', 'Beauty', 'Toys']
locations = ['Paris', 'Lyon', 'Marseille', 'Bordeaux', 'Lille']

df = pd.DataFrame({
    'Date': np.random.choice(dates, n_rows),
    'Category': np.random.choice(categories, n_rows),
    'Location': np.random.choice(locations, n_rows),
    'Quantity': np.random.randint(1, 10, n_rows),
    'Price': np.round(np.random.uniform(10, 500, n_rows), 2),
    'Rating': np.round(np.random.uniform(1, 5, n_rows), 1),
    'Discount_Applied': np.random.choice([True, False], n_rows, p=[0.3, 0.7])
})

# Feature engineering: Revenue
df['Revenue'] = df['Quantity'] * df['Price']

# Save to CSV
df.to_csv('EDA/sales_data.csv', index=False)
print("Dataset 'EDA/sales_data.csv' generated successfully.")
