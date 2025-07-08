from minio import Minio
from os import environ, getenv

MINIO_ACCESS_KEY = getenv('MINIO_ACCESS_KEY')
MINIO_SECRET_KEY = getenv('MINIO_SECRET_KEY')
MINIO_API_HOST = getenv('MINIO_ENDPOINT')
MINIO_BUCKET_URL = getenv('MINIO_BUCKET_URL')
DATA = getenv('DATA')
print(getenv('SSL_CERT_FILE'))
# environ["SSL_CERT_FILE"] = getenv('MINIO_CERT_PATH')
BUCKET = 'datasets'

def download_data_with_creds():
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

def download_data_without_creds():
    client = Minio(
    MINIO_API_HOST,
    secure=True
    )
    data = client.fget_object(
        BUCKET,
        f'data/{DATA}',
        f'test-data/{DATA}'
    )

download_data_without_creds()


