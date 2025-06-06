from minio import Minio
from os import environ, getenv


MINIO_ACCESS_KEY = getenv('MINIO_ACCESS_KEY')
MINIO_SECRET_KEY = getenv('MINIO_SECRET_KEY')
MINIO_API_HOST = getenv('MINIO_ENDPOINT')
DATA = getenv('DATA')
environ["SSL_CERT_FILE"] = getenv('MINIO_CERT_PATH', '')
BUCKET = 'datasets'

def download_data():
    client = Minio(
        MINIO_API_HOST,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=True,
        cert_check=True
    )
    data = client.fget_object(
        BUCKET,
        f'data/{DATA}',
        f'test-data/{DATA}'
    )
    return

download_data()

