import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder

N_CLUSTERS = 5
LINUX_FILE = '/Study/ASP/main/data/loghub - linux/Linux.log'
NGINX_FILE = '/Study/ASP/main/data/server logs - suspicious/CIDDS-001-external-week1.csv'
FEATURE_VECTOR = {'static': ['object', 'subject', 'environment', 'agent', 'transaction', 'source'],
                  'semi static': ['net', 'result code', 'log level', 'result', 'tag', 'label'],
                  'dynamic': ['timestamp', 'traceback', 'action', 'protocol', 'auth data', 'property']}


def get_linux_data(file_path):
    return pd.read_csv(file_path, sep='\n', names=['Log_contents'], encoding='unicode_escape')


def get_nginx_data(file_path):
    return pd.read_csv(file_path, sep=',', encoding='unicode_escape')


def preprocess_data(data):
    label_encoder = LabelEncoder()
    for c in data.columns:
        if data[c].dtype == 'object':
            encoded = label_encoder.fit_transform(data[c])
            data[c] = encoded
    return data


def k_means_clustering(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(data)
    labels = kmeans.labels_
    silhouette_avg = silhouette_score(data, labels)
    return silhouette_avg


def enrich_nginx_data(data):
    drop_features = data.columns[(data == 0).all()].tolist()
    print(f'Dropping features: {drop_features}')
    data = data.drop(columns=drop_features)
    print(data.head())
    # data.insert(4, FEATURE_VECTOR['semi static'][0], )
    data[FEATURE_VECTOR['semi static'][0]] = data['Src IP Addr'].astype(
        str) + ':' + data['Src Pt'].astype(str) + '-' + data['Dst IP Addr'].astype(str) + ':' + data['Dst Pt'].astype(str)
    return data

def main():
    linux_data = get_linux_data(LINUX_FILE)
    nginx_data = get_nginx_data(NGINX_FILE)
    nginx_data_a = preprocess_data(nginx_data)
    metric_a = k_means_clustering(nginx_data_a, N_CLUSTERS)
    print(f'Silhouette score for Nginx data A: {metric_a}')
    nginx_data = get_nginx_data(NGINX_FILE)
    nginx_data_b = enrich_nginx_data(nginx_data)
    nginx_data_b = preprocess_data(nginx_data_b)
    metric_b = k_means_clustering(nginx_data_b, N_CLUSTERS)
    print(f'Silhouette score for Nginx data B: {metric_b}')


if __name__ == '__main__':
    main()
