import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder

# Константы
N_CLUSTERS = 5  # Количество кластеров для алгоритма K-means
LINUX_FILE = '/Study/ASP/main/data/loghub - linux/Linux.log'  # Путь к файлу с логами Linux
NGINX_FILE = '/Study/ASP/main/data/server logs - suspicious/CIDDS-001-external-week1.csv'  # Путь к файлу с логами Nginx

# Вектор признаков для классификации
FEATURE_VECTOR = {
    'static': ['object', 'subject', 'environment', 'agent', 'transaction', 'source'],
    'semi static': ['net', 'result code', 'log level', 'result', 'tag', 'label'],
    'dynamic': ['timestamp', 'traceback', 'action', 'protocol', 'auth data', 'property']
}


def get_linux_data(file_path: str) -> pd.DataFrame:
    """Загрузка данных логов Linux из файла.
    
    Аргументы:
        file_path (str): Путь к файлу с логами
        
    Возвращает:
        pd.DataFrame: DataFrame с содержимым логов
    """
    return pd.read_csv(file_path, sep='\n', names=['Log_contents'], encoding='unicode_escape')


def get_nginx_data(file_path: str) -> pd.DataFrame:
    """Загрузка данных логов Nginx из файла.
    
    Аргументы:
        file_path (str): Путь к файлу с логами
        
    Возвращает:
        pd.DataFrame: DataFrame с содержимым логов
    """
    return pd.read_csv(file_path, sep=',', encoding='unicode_escape')


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """Предварительная обработка данных.
    
    Кодирует категориальные признаки с помощью LabelEncoder.
    
    Аргументы:
        data (pd.DataFrame): Исходные данные для обработки
        
    Возвращает:
        pd.DataFrame: Обработанные данные с закодированными признаками
    """
    label_encoder = LabelEncoder()
    for c in data.columns:
        if data[c].dtype == 'object':
            encoded = label_encoder.fit_transform(data[c])
            data[c] = encoded
    return data


def k_means_clustering(data: pd.DataFrame, n_clusters: int) -> float:
    """Кластеризация данных методом K-means.
    
    Аргументы:
        data (pd.DataFrame): Данные для кластеризации
        n_clusters (int): Количество кластеров
        
    Возвращает:
        float: Средний silhouette score для полученных кластеров
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(data)
    labels = kmeans.labels_
    silhouette_avg = silhouette_score(data, labels)
    return silhouette_avg


def enrich_nginx_data(data: pd.DataFrame) -> pd.DataFrame:
    """Обогащение данных Nginx новыми признаками.
    
    Удаляет пустые признаки и добавляет составной признак из IP-адресов и портов.
    
    Аргументы:
        data (pd.DataFrame): Исходные данные Nginx
        
    Возвращает:
        pd.DataFrame: Обогащенные данные
    """
    # Удаление пустых признаков
    drop_features = data.columns[(data == 0).all()].tolist()
    print(f'Удаляемые признаки: {drop_features}')
    data = data.drop(columns=drop_features)
    print(data.head())
    
    # Добавление составного признака
    data[FEATURE_VECTOR['semi static'][0]] = data['Src IP Addr'].astype(
        str) + ':' + data['Src Pt'].astype(str) + '-' + data['Dst IP Addr'].astype(str) + ':' + data['Dst Pt'].astype(str)
    return data


def main():
    """Основная функция выполнения скрипта."""
    # Загрузка данных
    linux_data = get_linux_data(LINUX_FILE)
    nginx_data = get_nginx_data(NGINX_FILE)
    
    # Вариант A: базовая предобработка
    nginx_data_a = preprocess_data(nginx_data)
    metric_a = k_means_clustering(nginx_data_a, N_CLUSTERS)
    print(f'Метрика silhouette для Nginx данных (вариант A): {metric_a}')
    
    # Вариант B: обогащение данных + предобработка
    nginx_data = get_nginx_data(NGINX_FILE)
    nginx_data_b = enrich_nginx_data(nginx_data)
    nginx_data_b = preprocess_data(nginx_data_b)
    metric_b = k_means_clustering(nginx_data_b, N_CLUSTERS)
    print(f'Метрика silhouette для Nginx данных (вариант B): {metric_b}')


if __name__ == '__main__':
    main()