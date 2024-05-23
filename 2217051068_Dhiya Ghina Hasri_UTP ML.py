# library
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# URL data csv
url = "https://raw.githubusercontent.com/jhnwlkn/Machine-Leaning/main/customer_support_data.csv"
# Tipe data kolom csv
dtype_dict = {
    'Unique id': str,
    'channel_name': str,
    'category': str,
    'Sub-category': str,
    'Customer Remarks': str,
    'Order_id': str,
    'order_date_time': str,
    'Issue_reported at': str,
    'issue_responded': str,
    'Survey_response_Date': str,
    'Customer_City': str,
    'Product_category': str,
    'Item_price': float,
    'connected_handling_time': float,
    'Agent_name': str,
    'Supervisor': str,
    'Manager': str,
    'Tenure Bucket': str,
    'Agent Shift': str,
    'CSAT Score': float
}
data = pd.read_csv(url, dtype=dtype_dict)

# Proses data
data = data.fillna(0)

# Menentukan fitur yang akan digunakan untuk clustering
X = data.iloc[:, [12, 13]].values

# Menentukan jumlah cluster
kmeans = KMeans(n_clusters=3, random_state=42)

# Proses clustering
clusters = kmeans.fit_predict(X)

# Menambahkan kolom cluster ke dalam dataframe
data['Cluster'] = clusters

# Evaluasi model menggunakan metode silhouette score
silhouette_avg = silhouette_score(X, clusters)
print("Silhouette Score : ", silhouette_avg)

# Menampilkan hasil clustering
print(data.head())
