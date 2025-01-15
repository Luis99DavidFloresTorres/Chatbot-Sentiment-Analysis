import pandas as pd
import boto3
class PreProcess:
    def __init__(self):
    def upload(self, local_file_path, s3_filename, bucket_name):

        s3 = boto3.client('s3')
        try:
            s3.upload_file(local_file_path,bucket_name,s3_filename)
            print(f"file {local_file_path} upload success in {bucket_name}/{s3_filename}")
        except Exception as e:
            print(f"Error al subir el archivo: {e}")
    def download(self, bucket_name, s3_file_name, local_file_path):
        s3 = boto3.client('s3')
        try:
            df = s3.download_file(bucket_name,s3_file_name, local_file_path)
            print(df)
            print(f"file {local_file_path} download success in {bucket_name}/{s3_file_name}")
        except Exception as e:
            print(f"Error to download the file: {e}")
        return pd.read_csv(df)
    def processText(self, data):
        print(data)