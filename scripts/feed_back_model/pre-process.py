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
                #inp = inp+' \nbot:'
                sentiment = df.loc[i-1, ['context']].apply(self.clean_text).item()
                text = df.loc[i-1, ['utterance']].apply(self.clean_text).item()
                inp = inp + ' bot: ' + text
               #structure = {'input': inp, 'conversation': text, 'sentiment': self.convert_sentiment(sentiment)}
                structure = {'input':inp, 'sentiment': self.convert_sentiment(sentiment)}
                dataset_preprocees.append(structure)
                text = df.loc[i, ['utterance']].apply(self.clean_text).item()
                inp =' human:' + text + ' \n'
            else:
                text = df.loc[i, ['utterance']].apply(self.clean_text).item()

                if i<len(df)-1 and df.loc[i+1, ['utterance_idx']].item()!=1:
                    if flagHuman:
                        inp=inp+' human:'+text+' \n'
                        flagHuman = False
                    else:
                        inp=inp+' bot:'+text+ ' \n'
                        flagHuman = True
                if i == len(df)-1:

                    sentiment = df.loc[i, ['context']].apply(self.clean_text).item()
                    text = df.loc[i , ['utterance']].apply(self.clean_text).item()
                    inp = inp + ' bot:' +text
                    #structure ={'input': inp, 'conversation': text, 'sentiment': self.convert_sentiment(sentiment)}
                    structure = {'input':inp, 'sentiment': self.convert_sentiment(sentiment)}
                    dataset_preprocees.append(structure)
        export = pd.DataFrame(dataset_preprocees)
        export.to_csv('../../dataset_empathetic/empatheticdialogues/train_dataset.csv', index=False)
        self.upload(local_file_path,s3_file_name,bucket_name)
    def clean_text(self, text):
        return str(text).replace("_comma_", ",").strip()
    def convert_sentiment_number(self,sentiment):
        jsn = {0:'sentimental',1:'afraid',2:'proud',3:'faithful',4:'terrified',
        5:'joyful',6:'angry',7:'sad',8:'jealous',9:'grateful',10:'prepared',
               11:'embarrassed',12:'excited',13:'annoyed',14:'lonely',15:'ashamed'
        ,16:'guilty',17:'surprised',18:'nostalgic',19:'confident',20:'furious',
               21:'disappointed',22:'caring',23:'trusting',24:'disgusted',25:'anticipating',
               26:'anxious',27:'hopeful',28:'content',29:'impressed',30:'apprehensive',31:'devastated'}
        return jsn[sentiment]
    def convert_sentiment(self,sentiment):
        jsn = {'sentimental':0,'afraid':1,'proud':2,'faithful':3,'terrified':4,
        'joyful':5,'angry':6,'sad':7,'jealous':8,'grateful':9,'prepared':10,
               'embarrassed':11,'excited':12,'annoyed':13,'lonely':14,'ashamed':15
        ,'guilty':16,'surprised':17,'nostalgic':18,'confident':19,'furious':20,
               'disappointed':21,'caring':22,'trusting':23,'disgusted':24,'anticipating':25,
               'anxious':26,'hopeful':27,'content':28,'impressed':29,'apprehensive':30,'devastated':31}
        return jsn[sentiment]
if __name__ == '__main__':
    PreProcess = PreProcess()
    PreProcess.processText('mlopsluis',"dataset/train_dataset.csv",'E:/Chatbot Sentiment Analysis/Chatbot-Sentiment-Analysis/dataset_empathetic/empatheticdialogues/train_dataset.csv')
    #print("people_comma_".replace("_comma_",","))
    #PreProcess.download("mlopsluis","dataset/chatbotSentiment.csv","traindownload.csv")
    #PreProcess.upload("E:/Chatbot Sentiment Analysis/Chatbot-Sentiment-Analysis/dataset_empathetic/empatheticdialogues/train1.csv","dataset/chatbotSentiment.csv","mlopsluis")