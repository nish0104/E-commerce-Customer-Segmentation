import pandas as pd # type: ignore
from sklearn.cluster import KMeans # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
import matplotlib.pyplot as plt # type: ignore

# Load the dataset
df = pd.read_excel('C:\\Users\\vagha\\OneDrive\\Documents\\Online Retail.xlsx', sheet_name='Online Retail')

# Data Cleaning
df = df.dropna(subset=['CustomerID'])  # Remove rows without CustomerID
df = df[df['Quantity'] > 0]  # Keep only positive quantities
df = df[df['UnitPrice'] > 0]  # Keep only positive prices

# Feature Engineering: Create a total spending column
df['TotalSpent'] = df['Quantity'] * df['UnitPrice']

# Aggregation: Group by CustomerID to calculate Recency, Frequency, and Monetary value (RFM)
customer_data = df.groupby('CustomerID').agg({
    'InvoiceDate': 'max',   # Recency
    'InvoiceNo': 'count',   # Frequency
    'TotalSpent': 'sum'     # Monetary value
}).rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'TotalSpent': 'Monetary'})

# Calculate days since last purchase (Recency)
latest_date = df['InvoiceDate'].max()
customer_data['Recency'] = (latest_date - customer_data['Recency']).dt.days

# Standardize the RFM values
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(customer_data)

# K-means Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
customer_data['Cluster'] = kmeans.fit_predict(rfm_scaled)

# Visualize Clusters
plt.scatter(customer_data['Frequency'], customer_data['Monetary'], c=customer_data['Cluster'], cmap='viridis')
plt.title('Customer Clusters')
plt.xlabel('Frequency')
plt.ylabel('Monetary Value')
plt.colorbar(label='Cluster')
plt.show()

# Save the results
customer_data.to_csv('customer_segments.csv', index=True)
