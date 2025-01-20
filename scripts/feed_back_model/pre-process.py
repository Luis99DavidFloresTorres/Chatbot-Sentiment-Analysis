import pandas as pd
import boto3
class PreProcess:
    def __init__(self):
        print("funciona")
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
            s3.download_file(bucket_name,s3_file_name, local_file_path)
            print(f"file {local_file_path} download success from {bucket_name}/{s3_file_name}")
        except Exception as e:
            print(f"Error to download the file: {e}")
    def processText(self, bucket_name,s3_file_name,local_file_path):
        df = pd.read_csv('../../dataset_empathetic/empatheticdialogues/traindownload.csv', on_bad_lines='skip')
        dataset_preprocees = []
        inp=""
        flagHuman = True
        for i in range(0,len(df)):

            txt_idx = df.loc[i, ['utterance_idx']].item()

            if txt_idx==1 and i!=0:
                inp = inp+' /nbot:'
                sentiment = df.loc[i-1, ['context']].apply(self.clean_text).item()
                text = df.loc[i-1, ['utterance']].apply(self.clean_text).item()
                structure = f'input: {inp}, conversation: {text}, sentiment: {sentiment}'
                dataset_preprocees.append(structure)
                text = df.loc[i, ['utterance']].apply(self.clean_text).item()
                inp =' human:' + text
            else:
                text = df.loc[i, ['utterance']].apply(self.clean_text).item()

                if i<len(df)-1 and df.loc[i+1, ['utterance_idx']].item()!=1:
                    if flagHuman:
                        inp=inp+' human:'+text
                        flagHuman = False
                    else:
                        inp=inp+' bot:'+text
                        flagHuman = True
                if i == len(df)-1:
                    inp = inp + ' /nbot:'
                    sentiment = df.loc[i, ['context']].apply(self.clean_text).item()
                    text = df.loc[i , ['utterance']].apply(self.clean_text).item()
                    structure =f'input: {inp}, conversation: {text}, sentiment: {sentiment}'
                    dataset_preprocees.append(structure)
        return dataset_preprocees
    def clean_text(self, text):
        return str(text).replace("_comma_", ",").strip()
if __name__ == '__main__':
    PreProcess = PreProcess()
    PreProcess.processText('.',',','.')
    #print("people_comma_".replace("_comma_",","))
    #PreProcess.download("mlopsluis","dataset/chatbotSentiment.csv","traindownload.csv")
    #PreProcess.upload("E:/Chatbot Sentiment Analysis/Chatbot-Sentiment-Analysis/dataset_empathetic/empatheticdialogues/train1.csv","dataset/chatbotSentiment.csv","mlopsluis")