from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_sequence
import random, re, os, numpy as np
from utils_audio import make_batch_roberta, make_batch_bert, make_batch_gpt
    
class MELD_loader(Dataset):
    def __init__(self, txt_file, dataclass):
        self.dialogs = []
        #audio_feature
        self.audio_feature = []
        csv_path_train = '/import/c4dm-datasets/jl007/Data/Meld/emo_csv_384/train_384_csv'
        csv_files_train = os.listdir(csv_path_train)
        csv_files_train.sort(key=lambda x: (int(x.split('_')[0][3:]), int(x.split('_')[1][3:-4])))

        for csv_file in csv_files_train:
            with open(csv_path_train + '/' + csv_file) as f:
                last_line = f.readlines()[-1]  
            feature = last_line.split(",")
            feature = np.array(feature[1:-1], dtype="float64").tolist()  
            k, nums = re.findall(r"\d+\d*", csv_file)
            self.audio_feature.append(feature)
        
        f = open(txt_file, 'r')
        dataset = f.readlines()
        f.close()
        
        temp_speakerList = []
        context = []
        context_speaker = []
        self.speakerNum = []
        # 'anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise'
        emodict = {'anger': "anger", 'disgust': "disgust", 'fear': "fear", 'joy': "joy", 'neutral': "neutral", 'sadness': "sad", 'surprise': 'surprise'}
        self.sentidict = {'positive': ["joy"], 'negative': ["anger", "disgust", "fear", "sadness"], 'neutral': ["neutral", "surprise"]}
        self.emoSet = set()
        self.sentiSet = set()
        for i, data in enumerate(dataset):
            if i < 2:
                continue
            if data == '\n' and len(self.dialogs) > 0:
                self.speakerNum.append(len(temp_speakerList))
                temp_speakerList = []
                context = []
                context_speaker = []
                continue
            speaker, utt, emo, senti = data.strip().split('\t')
            context.append(utt)
            if speaker not in temp_speakerList:
                temp_speakerList.append(speaker)
            speakerCLS = temp_speakerList.index(speaker)
            context_speaker.append(speakerCLS)
            
            self.dialogs.append([context_speaker[:], context[:], emodict[emo], senti])
            self.emoSet.add(emodict[emo])
            self.sentiSet.add(senti)

        self.emoList = sorted(self.emoSet)  
        self.sentiList = sorted(self.sentiSet)
        if dataclass == 'emotion':
            self.labelList = self.emoList
        else:
            self.labelList = self.sentiList        
        self.speakerNum.append(len(temp_speakerList))

       
        
    def __len__(self):
        return len(self.dialogs)

    def __getitem__(self, idx):
        return self.dialogs[idx], self.labelList, self.sentidict, self.audio_feature[idx]
    
    
class Emory_loader(Dataset):
    def __init__(self, txt_file, dataclass):
        self.dialogs = []
        
        f = open(txt_file, 'r')
        dataset = f.readlines()
        f.close()
        """sentiment"""
        # 'Joyful', 'Mad', 'Neutral', 'Peaceful', 'Powerful', 'Sad', 'Scared'
        pos = ['Joyful', 'Peaceful', 'Powerful']
        neg = ['Mad', 'Sad', 'Scared']
        neu = ['Neutral']
        emodict = {'Joyful': "joy", 'Mad': "mad", 'Peaceful': "peaceful", 'Powerful': "powerful", 'Neutral': "neutral", 'Sad': "sad", 'Scared': 'scared'}
        self.sentidict = {'positive': pos, 'negative': neg, 'neutral': neu}
        temp_speakerList = []
        context = []
        context_speaker = []        
        self.speakerNum = []
        self.emoSet = set()
        self.sentiSet = set()
        for i, data in enumerate(dataset):
            if data == '\n' and len(self.dialogs) > 0:
                self.speakerNum.append(len(temp_speakerList))
                temp_speakerList = []
                context = []
                context_speaker = []
                continue
            speaker, utt, emo = data.strip().split('\t')
            context.append(utt)
            
            if emo in pos:
                senti = "positive"
            elif emo in neg:
                senti = "negative"
            elif emo in neu:
                senti = "neutral"
            else:
                print('ERROR emotion&sentiment')
                
            if speaker not in temp_speakerList:
                temp_speakerList.append(speaker)
            speakerCLS = temp_speakerList.index(speaker)
            context_speaker.append(speakerCLS)
            
            self.dialogs.append([context_speaker[:], context[:], emodict[emo], senti])
            self.emoSet.add(emodict[emo])
            self.sentiSet.add(senti)
            
        self.emoList = sorted(self.emoSet)
        self.sentiList = sorted(self.sentiSet)
        if dataclass == 'emotion':
            self.labelList = self.emoList
        else:
            self.labelList = self.sentiList        
        self.speakerNum.append(len(temp_speakerList))
        
    def __len__(self):
        return len(self.dialogs)

    def __getitem__(self, idx):
        return self.dialogs[idx], self.labelList, self.sentidict
    
    
class IEMOCAP_loader(Dataset):
    def __init__(self, txt_file, dataclass):
        self.dialogs = []
        
        f = open(txt_file, 'r')
        dataset = f.readlines()
        f.close()
        
        temp_speakerList = []
        context = []
        context_speaker = []
        self.speakerNum = []
        pos = ['ang', 'exc', 'hap']
        neg = ['fru', 'sad']
        neu = ['neu']
        emodict = {'ang': "angry", 'exc': "excited", 'fru': "frustrated", 'hap': "happy", 'neu': "neutral", 'sad': "sad"}
        self.sentidict = {'positive': pos, 'negative': neg, 'neutral': neu}
        # use: 'hap', 'sad', 'neu', 'ang', 'exc', 'fru'
        # discard: disgust, fear, other, surprise, xxx        
        self.emoSet = set()
        self.sentiSet = set()
        for i, data in enumerate(dataset):
            if data == '\n' and len(self.dialogs) > 0:
                self.speakerNum.append(len(temp_speakerList))
                temp_speakerList = []
                context = []
                context_speaker = []
                continue
            speaker = data.strip().split('\t')[0]
            utt = ' '.join(data.strip().split('\t')[1:-1])
            emo = data.strip().split('\t')[-1]
            context.append(utt)
            
            if emo in pos:
                senti = "positive"
            elif emo in neg:
                senti = "negative"
            elif emo in neu:
                senti = "neutral"
            else:
                print('ERROR emotion&sentiment')                        
            
            if speaker not in temp_speakerList:
                temp_speakerList.append(speaker)
            speakerCLS = temp_speakerList.index(speaker)
            context_speaker.append(speakerCLS)
            
            self.dialogs.append([context_speaker[:], context[:], emodict[emo], senti])
            self.emoSet.add(emodict[emo])
        
        self.emoList = sorted(self.emoSet)   
        self.sentiList = sorted(self.sentiSet)
        if dataclass == 'emotion':
            self.labelList = self.emoList
        else:
            self.labelList = self.sentiList        
        self.speakerNum.append(len(temp_speakerList))
        
    def __len__(self):
        return len(self.dialogs)

    def __getitem__(self, idx):
        return self.dialogs[idx], self.labelList, self.sentidict
    
class DD_loader(Dataset):
    def __init__(self, txt_file, dataclass):
        self.dialogs = []
        
        f = open(txt_file, 'r')
        dataset = f.readlines()
        f.close()
        
        temp_speakerList = []
        context = []
        context_speaker = []
        self.speakerNum = []      
        self.emoSet = set()
        self.sentiSet = set()
        # {'anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise'}
        pos = ['happiness']
        neg = ['anger', 'disgust', 'fear', 'sadness']
        neu = ['neutral', 'surprise']
        emodict = {'anger': "anger", 'disgust': "disgust", 'fear': "fear", 'happiness': "happy", 'neutral': "neutral", 'sadness': "sad", 'surprise': "surprise"}
        self.sentidict = {'positive': pos, 'negative': neg, 'neutral': neu}
        for i, data in enumerate(dataset):
            if data == '\n' and len(self.dialogs) > 0:
                self.speakerNum.append(len(temp_speakerList))
                temp_speakerList = []
                context = []
                context_speaker = []
                continue
            speaker = data.strip().split('\t')[0]
            utt = ' '.join(data.strip().split('\t')[1:-1])
            emo = data.strip().split('\t')[-1]
            
            if emo in pos:
                senti = "positive"
            elif emo in neg:
                senti = "negative"
            elif emo in neu:
                senti = "neutral"
            else:
                print('ERROR emotion&sentiment')                
            
            context.append(utt)
            if speaker not in temp_speakerList:
                temp_speakerList.append(speaker)
            speakerCLS = temp_speakerList.index(speaker)
            context_speaker.append(speakerCLS)
            
            self.dialogs.append([context_speaker[:], context[:], emodict[emo], senti])
            self.emoSet.add(emodict[emo])
        
        self.emoList = sorted(self.emoSet)   
        self.sentiList = sorted(self.sentiSet)
        if dataclass == 'emotion':
            self.labelList = self.emoList
        else:
            self.labelList = self.sentiList        
        self.speakerNum.append(len(temp_speakerList))

    def __len__(self):
        return len(self.dialogs)

    def __getitem__(self, idx):
        return self.dialogs[idx], self.labelList, self.sentidict, self.audio_feature[idx]


if __name__ == '__main__':
    dataset = 'MELD'
    dataType = 'multi'
    data_path = './dataset/MELD/'+dataType+'/'
    train_path = data_path + dataset+'_train.txt'
    dataclass = 'emotion'

    #dataloader
    batch_size = 1
    model_type = 'roberta'
    if 'roberta' in model_type:
        make_batch = make_batch_roberta
    elif model_type == 'bert-large-uncased':
        make_batch = make_batch_bert
    else:
        make_batch = make_batch_gpt  
        
    train_dataset = MELD_loader(txt_file = train_path, dataclass = dataclass)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=make_batch)

    print(type(train_dataloader))
    print('train_dataloader len', len(train_dataloader))
    for i_batch, data in enumerate(train_dataloader):            
            batch_input_tokens, batch_labels, batch_speaker_tokens, batch_audio = data
            print(i_batch, 'batch_input_tokens type', type(batch_input_tokens), batch_input_tokens)
            print(i_batch, 'batch_labels type', type(batch_labels), batch_labels)
            print(i_batch, 'batch_speaker_tokens type', type(batch_speaker_tokens))
            print(i_batch, 'batch_audio type', type(batch_audio), batch_audio, len(batch_audio))
            i_batch == 1
            i_batch += 1
            if  i_batch == 2:
                break

    
    # for i, dialogs, labelList, sentidict in train_dataloader:
    #     print('dialogs type', type(dialogs))
    #     print('labelList type', type(labelList))
    #     print('sentidict type', type(sentidict))

    #     i = 0
    #     i += 1
    #     if i == 3:
    #         break

