
import boto3

if __name__ == '__main__':
    s3 = boto3.client('s3')
    s3.copy_object(
        Bucket='mlopsluis',
        CopySource={'Bucket': 'mlopsluis', 'Key': 'dataset/train_dataset.csv'},
        Key='pruebacopy.csv',
    )
