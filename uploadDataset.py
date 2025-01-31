from scripts.feed_back_model.preprocess import PreProcess

if __name__ == '__main__':
    preprocess = PreProcess()
    preprocess.upload('train_dataset.csv','dataset/train_dataset.csv','mlopsluis')