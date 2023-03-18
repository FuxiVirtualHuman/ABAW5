import os
import pickle
from tqdm import tqdm
import pandas as pd
import mxnet as mx
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torch
import time
# from transformers import AutoTokenizer, AutoModel, AutoConfig, Wav2Vec2Processor 
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from scipy.io import wavfile
import soundfile as sf
# import torchaudio
import cv2
import random
from scipy.special import softmax
from scipy.stats import pearsonr
# import librosa

COUNTRY = {'United States':0, 'South Africa':1}

# class DeBERTA:
#   def __init__(self, model_name="microsoft/deberta-v3-large") -> None:
#     self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
#     self.model = AutoModel.from_pretrained('microsoft/deberta-v3-large')
#     print('initial DeBERTa model:', model_name)
  
#   def run(self, text):
#     inputs = self.tokenizer(text, return_tensors="pt")
#     with torch.no_grad():
#         logits = self.model(**inputs)['last_hidden_state']
#     return logits

class Dataset_ERI(Dataset):
    def __init__(self, root, split, img_max=100, transforms=None, use_img2rec=False) -> None:
        # root = '/data/data/ABAW5/challenge4/'
        self.root = root
        self.image_root = os.path.join(root,'crop_face', split+'npy')
        self.audio_root = os.path.join(root,f'{split}_hubert')
        self.transforms = transforms
        self.use_img2rec = use_img2rec
        self.features = {}
        self.vid_list = self.load_annos(split)
        self.img_max = img_max
        self._load_txt(split)
        # self._load_audio_feature(split)
        
        # self.img_idx, self.img_rec = self._load_images(split)
        self.EMOTION_NAME = ['Adoration', 'Amusement', 'Anxiety', 'Disgust', 'Empathic-Pain', 'Fear']
        print(f'loaded {split} data, total {len(self.vid_list)}')
        
    def load_annos(self, split):
        df = pd.read_csv(os.path.join(self.root, 'data_info.csv'))
        # row_number
        rows_n = len(df['File_ID'])
        vid_list = []
        for i in range(rows_n):
            if df['Split'][i].lower() == split:
                vid_list.append(df['File_ID'][i].strip('[]'))
                self.features[df['File_ID'][i].strip('[]')] = {
                    'Adoration': df['Adoration'][i],
                    'Amusement': df['Amusement'][i],
                    'Anxiety': df['Anxiety'][i],
                    'Disgust': df['Disgust'][i],
                    'Empathic-Pain': df['Empathic-Pain'][i],
                    'Fear': df['Fear'][i],
                    'Age': df['Age'][i],
                    'Country': df['Country'][i],
                }
        return vid_list
        
    def _load_txt(self, split):
        print('loading txt and feature...')
        with open(os.path.join(self.root, split+'_text.txt'),'r') as f:
            data = f.readlines()
        for line in data:
            vid_id, text = line.strip().split('.wav ')
            self.features[vid_id]['sentence'] = text.replace('sil ', '').replace(' sil ', '').replace('sil', '').capitalize().strip(' ')
            
        file_feature = os.path.join(self.root, split+'_debert.pkl')
        debert_feature = {}
        if not os.path.exists(file_feature):   
            deberta = DeBERTA()
            feature_blank = deberta.run('').cpu().numpy()     
            for line in tqdm(data, total=len(data)):
                vid_id, _ = line.strip().split('.wav ')
                text = self.features[vid_id]['sentence']                
                if text == '':
                    txt_feature = feature_blank
                else:
                    txt_feature = deberta.run(self.features[vid_id]['sentence']).cpu().numpy()
                debert_feature[vid_id] = txt_feature
            with open(file_feature, 'wb') as f:
                pickle.dump(debert_feature, f)
        
        with open(file_feature, 'rb') as f:
            features = pickle.load(f)
        for line in data:
            vid_id, _ = line.strip().split('.wav ')
            self.features[vid_id]['sentence_feature'] = features[vid_id]
        
        return
    
    def _load_audio_feature(self, split):
        with open(os.path.join(self.root, split+'_hubert.pkl'),'rb') as f:
            data = pickle.load(f)
        for key, value in data.items():
            self.features[key]['hubert'] = value
        return

    def _load_images(self, split):
        print('loading image name...')
        path_lst = os.path.join(self.root, 'crop_face', split+'.lst')
        with open(path_lst, 'r') as f:
            data_lst = f.readlines()
        img_idx = {d.strip().split('\t')[-1]:d.strip().split('\t')[0] for d in data_lst}
        img_names = list(img_idx.keys())
        img_names.sort()
        for line in img_names:
            vid_id = line.split('/')[0]
            if 'images' not in self.features[vid_id]:
                self.features[vid_id]['images'] = [line]
            else:
                self.features[vid_id]['images'].append(line)
        img_rec = None
        if self.use_img2rec:
            for vid_id in self.features.keys():
                self.features[vid_id]['images'] = self.features[vid_id]['images']
            # load cache images data
            path_rec = path_lst.replace('.lst', '.rec')
            path_idx = path_lst.replace('.lst', '.idx')

            img_rec = mx.recordio.MXIndexedRecordIO(path_idx, path_rec, 'r')
        return img_idx, img_rec
    
    def _read_image_img2rec(self, imgname):
        # return: PIL Image
        idx = int(self.img_idx[imgname])
        s = self.img_rec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        img = mx.image.imdecode((img)).asnumpy()
        img = Image.fromarray(np.uint8(img))
        return img
    
    def __getitem__(self, index):
        vid_id = self.vid_list[index]
        # vid_id = '11526'
        data = self.features[vid_id]
        try:
            with open(os.path.join(self.image_root+'_pkl', vid_id+'.pkl'), 'rb') as f:
                videos = pickle.load(f)
        except:
            print(vid_id)
        # video = data['images']
        frames = videos
        if len(frames) < self.img_max:
            frames = frames +frames
        img_step = len(frames)//self.img_max
        frames = frames[::-1][:img_step * self.img_max:img_step][::-1]
 
        imgs = [] 
        
        # for imgname in frames:
        #     if self.use_img2rec:
        #         img = self._read_image_img2rec(imgname)
        #     else:
        #         try:
        #             img = videos[int(imgname.split('.')[0].split('/')[-1])]
        #         except:
        #             print(vid_id)
                # try:
                #     img = Image.open(os.path.join(self.image_root, imgname)).convert('RGB')
                # except:
                #     img = Image.open(os.path.join(self.image_root, self.features[0]['images'][0])).convert('RGB')
        for img in frames:
            if self.transforms:
                try:
                    img = self.transforms(img)
                except:
                    print(vid_id)
            imgs.append(img)
            
        label = np.array([data[e] for e in self.EMOTION_NAME])
        #
        with open(os.path.join(self.audio_root, vid_id),'rb') as f:
            hubert = pickle.load(f)
        deberta = data['sentence_feature']

        return {'imgs': torch.stack(imgs),
                'label': label,
                'audio': hubert,
                'text': deberta,
                }
     
    def __len__(self):
        return len(self.vid_list)

class Dataset_ERI_MAE_audio(Dataset):
    def __init__(self, root, split, img_max=100, transforms=None, fold_i=0,use_mae_feature=False,
                 use_visual=False, use_audio=False, use_text=False, use_dual=False, shuffle_frame=False) -> None:
        # root = '/data/data/ABAW5/challenge4/'
        self.root = root
        self.image_root = os.path.join(root,'crop_face', split+'_npy')
        self.audio_root = os.path.join(root,f'{split}_hubertbase_npy')
        self.audio_root2 = os.path.join(root,f'{split}_wav2vec_npy')
        self.audio_root3 = os.path.join(root,f'{split}_vggish_npy')
        self.wav_root = f'/project/zhangwei/ABAW5/challenge4/{split}/wav'
        self.split = split
        self.transforms = transforms
        self.features = {}
        self.vid_list = self.load_annos(split)[:]
        # self._load_mae_feature(split)
        self.img_max = img_max
        self.use_visual = use_visual
        self.use_audio = use_audio
        self.use_text = use_text
        self.shuffle_frame = shuffle_frame
        self.use_mae_feature = use_mae_feature
        self.use_mfcc = True
        self._load_clip_annos(split)
        # self.vid_list = [v for v in self.vid_list if self.features[v]['Country'] == 1]
        self.EMOTION_NAME = ['Adoration', 'Amusement', 'Anxiety', 'Disgust', 'Empathic-Pain', 'Fear', 'Surprise', 'Country']

        print(f'loaded {split} data, total {len(self.vid_list)}')

    def _load_clip_annos(self, split):
        with open(os.path.join(self.root, f'clip_annotation_{split}.pkl'), 'rb') as f:
            data = pickle.load(f)
        for vidid in data.keys():
            if vidid in self.features:
                self.features[vidid]['clip'] = data[vidid]
        return
    
    def load_annos(self, split):
        df = pd.read_csv(os.path.join(self.root, 'data_info.csv'))
        # row_number
        rows_n = len(df['File_ID'])
        vid_list = []
        for i in range(rows_n):
            if df['Split'][i].lower() == split:
                vid_list.append(df['File_ID'][i].strip('[]'))
                self.features[df['File_ID'][i].strip('[]')] = {
                    'Adoration': df['Adoration'][i],
                    'Amusement': df['Amusement'][i],
                    'Anxiety': df['Anxiety'][i],
                    'Disgust': df['Disgust'][i],
                    'Empathic-Pain': df['Empathic-Pain'][i],
                    'Fear': df['Fear'][i],
                    'Surprise': df['Surprise'][i],
                    'Age': df['Age'][i],
                    'Country': COUNTRY[df['Country'][i]],
                }
                
        npz_file = f'/data/data/ABAW5/challenge4/AffectNet/{split}.npz'
        npz = np.load(npz_file)
        file_path = npz['file_relpath']
        # features = npz['feature']
        pred = npz['expression_pred']
        pred = softmax(pred, axis=1)
        EMOTIONS ={0: 'Neutral', 1: 'Happy', 2: 'Sad', 3: 'Surprise', 4: 'Fear', 5: 'Disgust', 6: 'Anger'}
        n = len(file_path)
        prev_i = 0
        for i in range(n-1):
            if file_path[i][:5] != file_path[i+1][:5]:
                self.features[file_path[i][:5]]['neutral'] = pred[prev_i:i+1, 0]
                prev_i = i + 1 
                
        # load_text_fea
        with open(os.path.join(self.root, f'{split}_debert.pkl'), 'rb') as f:
            data = pickle.load(f)
        for vid, fea in data.items():
            self.features[vid]['deberta'] = fea
        return vid_list



    def __getitem__(self, index):
        vid_id = self.vid_list[index]
        data = self.features[vid_id]
        label = np.array([data[e] for e in self.EMOTION_NAME])
        sample = {'label': label, 'visual':None, 'audio':None, 'text':None, 
                  'vid':vid_id, 'pretrained_feature':None}

        audio = np.load(os.path.join(self.root,f'{self.split}_hubertbase_npy', f'{vid_id}.npy')).squeeze(0)
        max_len = min(800, len(audio))
        step_n = len(audio)/max_len
        index = [int(i*step_n) for i in range(max_len-1)]
        sample['audio'] = np.array(audio[index])

        # audio = np.load(os.path.join(self.root,f'{self.split}_MFCC_npy', f'{vid_id}.npy')).transpose(1,0)
        # audio = np.stack([audio[i*8:(i+1)*8].flatten() for i in range(len(audio)//8)])
        # audio = np.reshape(audio, (-1,1024))
        # sample['audio'] = librosa.power_to_db(np.array(audio))/100.
        return sample
     
    def __len__(self):
        return len(self.vid_list)
   
class Dataset_ERI_MAE(Dataset):
    def __init__(self, root, split, img_max=100, transforms=None, fold_i=0,use_mae_feature=False,
                 use_visual=False, use_audio=False, use_text=False, use_dual=False, shuffle_frame=False) -> None:
        # root = '/data/data/ABAW5/challenge4/'
        self.root = root
        self.image_root = os.path.join(root,'crop_face', split+'_npy')
        self.audio_root = os.path.join(root,f'{split}_hubertbase_npy')
        self.audio_root2 = os.path.join(root,f'{split}_wav2vec_npy')
        self.audio_root3 = os.path.join(root,f'{split}_vggish_npy')
        self.wav_root = f'/project/zhangwei/ABAW5/challenge4/{split}/wav'
        self.split = split
        self.transforms = transforms
        self.features = {}
        self.vid_list = self.load_annos(split)[:]
        # self._load_mae_feature(split)
        self.img_max = img_max
        self.use_visual = use_visual
        self.use_audio = use_audio
        self.use_text = use_text
        self.shuffle_frame = shuffle_frame
        self.use_mae_feature = use_mae_feature
        self.use_mfcc = False
        # self.vid_list = self._resample_data() 
        # vid_list_resample = self._build_hard_set(split)
        # if use_dual:
        #     self.vid_list = vid_list_resample
        # if use_audio:
        #     # self.audio_processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
        #     bundle = torchaudio.pipelines.HUBERT_BASE
        #     self.audio_processor = bundle.get_model().cuda()

        # self.img_idx, self.img_rec = self._load_images(split)
        self._load_clip_annos(split)
        # self.vid_list = [v for v in self.vid_list if self.features[v]['Country'] == 1]
        self.EMOTION_NAME = ['Adoration', 'Amusement', 'Anxiety', 'Disgust', 'Empathic-Pain', 'Fear', 'Surprise', 'Country']
        
        # pcc = []
        # for i in range(len(self.EMOTION_NAME)):
        #     pcci = []
        #     for j in range(len(self.EMOTION_NAME)):
        #         pcci.append(pearsonr([self.features[s][self.EMOTION_NAME[i]]  for s in self.features],[self.features[s][self.EMOTION_NAME[j]]  for s in self.features])[0])
        #     pcc.append(pcci)
        #     print(pcci)
        # print(pcc)
        # self.pcc = np.array(pcc)
        print(f'loaded {split} data, total {len(self.vid_list)}')

    def _load_clip_annos(self, split):
        with open(os.path.join(self.root, f'clip_annotation_{split}.pkl'), 'rb') as f:
            data = pickle.load(f)
        for vidid in data.keys():
            if vidid in self.features:
                self.features[vidid]['clip'] = data[vidid]
        return
    
    def _build_hard_set(self, split, hard_ratio=0.15):
        # method from paper: Dual-branch Collaboration Network for Macro- and Micro-expression Spotting
        pred_root = '/data/Workspace/ABAW/code_ABAW5/test'
        pretrained = '20230220-124201_10'
        with open(os.path.join(pred_root, f'{pretrained}_{split}.pkl'), 'rb') as f:
            data = pickle.load(f)
        n = len(data['vid'])
        for i in range(n):
            self.features[data['vid'][i]]['pretrained_feature'] = data['preds'][i][:7]
        index = list(range(n))
        index.sort(key=lambda x: data['dist'][x], reverse=True)
        hard_index = index[:int(n*hard_ratio)]
        hard_vid = [data['vid'][i] for i in hard_index]
        orgin_vid = random.choices(data['vid'], k = int((1-hard_ratio)*n))
        resample_vid = hard_vid + orgin_vid
        return resample_vid
        
    def _resample_data(self):
        adoration = [v for v in self.vid_list if self.features[v]['Adoration'] > 0.5]
        surprise = [v for v in self.vid_list if self.features[v]['Surprise'] > 0.5]
        rest = [v for v in self.vid_list if v not in adoration+surprise]
        vid_list_resampled = adoration[::2] + surprise[::2] + rest
        return vid_list_resampled
    
    def load_annos(self, split):
        df = pd.read_csv(os.path.join(self.root, 'data_info.csv'))
        # row_number
        rows_n = len(df['File_ID'])
        vid_list = []
        for i in range(rows_n):
            if df['Split'][i].lower() == split:
                vid_list.append(df['File_ID'][i].strip('[]'))
                self.features[df['File_ID'][i].strip('[]')] = {
                    'Adoration': df['Adoration'][i],
                    'Amusement': df['Amusement'][i],
                    'Anxiety': df['Anxiety'][i],
                    'Disgust': df['Disgust'][i],
                    'Empathic-Pain': df['Empathic-Pain'][i],
                    'Fear': df['Fear'][i],
                    'Surprise': df['Surprise'][i],
                    'Age': df['Age'][i],
                    'Country': COUNTRY[df['Country'][i]],
                }
                
        npz_file = f'/data/data/ABAW5/challenge4/AffectNet/{split}.npz'
        npz = np.load(npz_file)
        file_path = npz['file_relpath']
        # features = npz['feature']
        pred = npz['expression_pred']
        pred = softmax(pred, axis=1)
        EMOTIONS ={0: 'Neutral', 1: 'Happy', 2: 'Sad', 3: 'Surprise', 4: 'Fear', 5: 'Disgust', 6: 'Anger'}
        n = len(file_path)
        prev_i = 0
        for i in range(n-1):
            if file_path[i][:5] != file_path[i+1][:5]:
                self.features[file_path[i][:5]]['neutral'] = pred[prev_i:i+1, 0]
                prev_i = i + 1 
                
        # load_text_fea
        with open(os.path.join(self.root, f'{split}_debert.pkl'), 'rb') as f:
            data = pickle.load(f)
        for vid, fea in data.items():
            self.features[vid]['deberta'] = fea
        return vid_list

    def _load_mae_feature(self, split):
        npz_file = f'/project/mbw/mae/eval.AffectNet/9001.ABAW5.challenge4.{split}/eval.npz'
        npz = np.load(npz_file)
        file_path = npz['file_relpath']
        features = npz['feature']
        pred = npz['expression_pred']
        lenth = len(file_path)
        start = 0
        for i in range(1, lenth):
            if file_path[i][:5] != file_path[i-1][:5]:
                vid_id = file_path[i-1][:5]
                self.features[vid_id]['mae_feature'] = features[start:i]
                start = i
        vid_id = file_path[-1][:5]
        self.features[vid_id]['mae_feature'] = features[start:]
        return

    def _load_images(self, split):
        print('loading image name...')
        path_lst = os.path.join(self.root, 'crop_face', split+'.lst')
        with open(path_lst, 'r') as f:
            data_lst = f.readlines()
        img_idx = {d.strip().split('\t')[-1]:d.strip().split('\t')[0] for d in data_lst}
        img_names = list(img_idx.keys())
        img_names.sort()
        for line in img_names:
            vid_id = line.split('/')[0]
            if 'images' not in self.features[vid_id]:
                self.features[vid_id]['images'] = [line]
            else:
                self.features[vid_id]['images'].append(line)
        img_rec = None
        return img_idx, img_rec
    

    def __getitem__(self, index):
        vid_id = self.vid_list[index]
        data = self.features[vid_id]
        label = np.array([data[e] for e in self.EMOTION_NAME])
        sample = {'label': label, 'visual':None, 'audio':None, 'text':None, 
                  'vid':vid_id, 'pretrained_feature':None}

        if self.use_visual:
            frames = np.load(os.path.join(self.image_root, f'{vid_id}.npy'), allow_pickle=True)
            # if 'clip' in data and (data['clip'][1]-data['clip'][0])>=16 and data['clip'][1] < frames.shape[0]:
                # frames = frames[data['clip'][0]:data['clip'][1]]

            # 利用affectnet选图的方式
            # if 'neutral' in data:
            #     attention = 1 - np.array(data['neutral'])[:len(frames)]
            # else:
            #     attention = np.ones(len(frames))
            # action_indexes = np.where(attention>0.5)[0]
            # if action_indexes.shape[0] < 5: #如果全是中性，则取整个视频
            #     action_indexes = np.arange(0, len(frames))
            # if action_indexes.shape[0] < self.img_max: # 如果帧数不够，复制一下
            #     action_indexes = np.repeat(action_indexes, int(np.ceil(self.img_max/len(action_indexes))))
            # img_step = len(action_indexes)/float(self.img_max)
            # indexes = np.array([int(i*img_step+random.uniform(0, img_step)) for i in range(self.img_max)])
            # try:
            #     frames = frames[action_indexes[indexes]]
            # except:
            #     print(indexes)
                            
            # 正常的取图的方式。
            if len(frames) < self.img_max:
                frames = np.concatenate([frames, frames],axis=0)
            n = len(frames)
            img_step = len(frames)/float(self.img_max)
            indexes = np.array([int(i*img_step+random.uniform(0, img_step)) for i in range(self.img_max)])
            frames = frames[indexes]
            if self.shuffle_frame and np.random.randn() < 0.5:
                np.random.shuffle(frames)
            images = []
            for im in frames:
                images.append(Image.fromarray(im))
            if self.transforms:
                images =  [self.transforms(im) for im in images]
            sample['visual'] = torch.stack(images)
            # try:
            #     sample['pretrained_feature'] = data['pretrained_feature']
            # except:
            sample['pretrained_feature'] = np.ones(1)
            if 'neutral' in data and indexes[-1] < len(data['neutral']):
                sample['attention_map'] = 1-data['neutral'][indexes]
            else:
                sample['attention_map'] = np.ones(self.img_max)
            
            if self.use_mae_feature:
                
                sample['visual'] = np.load(os.path.join(self.root, f'{self.split}_mae', f'{vid_id}.npy'))[:400]
                # sample['visual'] = np.zeros((400, 768))

            # cv2.imwrite(f'/data/Downloads/{vid_id}.png',np.array(images[0]*255, dtype=np.uint8).transpose((1,2,0)))

        if self.use_audio:
            # wav, rate = sf.read(os.path.join(self.wav_root, f'{vid_id}.wav'))
            # audio = self.audio_processor(wav[:3200*100], sampling_rate=rate, return_tensors='pt').input_values
            # wav, rate = torchaudio.load(os.path.join(self.wav_root, f'{vid_id}.wav'))
            # audio = self.audio_processor.extract_features(wav.cuda())[0][-1]
            # sample['audio'] = np.array(audio.detach().numpy())
            if self.use_mfcc:
                audio = np.load(os.path.join(self.root,f'{self.split}_hubertbase_npy', f'{vid_id}.npy'))
                # audio = np.stack([audio[i*8:(i+1)*8].flatten() for i in range(len(audio)//8)])
                # audio = np.reshape(audio, (-1,1024))
                sample['audio'] = np.array(audio)
            else:
                audio = np.load(os.path.join(self.audio_root, f'{vid_id}.npy')).squeeze()
                # sample['audio'] = np.array(audio)[:,:800]
                max_len = min(800, len(audio))
                step_n = len(audio)/max_len
                index = [int(i*step_n) for i in range(max_len-1)]
                sample['audio'] = np.array(audio[index])

        if self.use_mae_feature:
            min_len = min(sample['visual'].shape[-2], sample['audio'].shape[-2])
            step_v, step_a = sample['visual'].shape[-2]/min_len, sample['audio'].shape[-2]/min_len
            sample['visual'] = sample['visual'][np.array([int(i*step_v) for i in range(min_len)])]
            sample['audio'] = sample['audio'][:,np.array([int(i*step_a) for i in range(min_len)])]
        
        if self.use_text:
            if 'deberta' in data:
                text = data['deberta']
            else:
                text = np.zeros((1, 2, 1024))
            sample['text'] = text
        return sample
     
    def __len__(self):
        return len(self.vid_list)

class Dataset_ERI_MAE_5split(Dataset):
    def __init__(self, root, split, img_max=100, transforms=None, fold_i=1, use_mae_feature=False,
                 use_visual=False, use_audio=False, use_text=False, use_dual=False, shuffle_frame=False) -> None:
        # root = '/data/data/ABAW5/challenge4/'
        self.root = root
        self.transforms = transforms
        self.features = {}
        self.vid_list = self.load_annos(split, fold_i)[:]
        self.img_max = img_max
        self.use_visual = use_visual
        self.use_audio = use_audio
        self.use_text = use_text
        self.shuffle_frame = shuffle_frame
        self.use_mae_feature = use_mae_feature
        self.split = split
        self.EMOTION_NAME = ['Adoration', 'Amusement', 'Anxiety', 'Disgust', 'Empathic-Pain', 'Fear', 'Surprise', 'Country']
        
        print(f'loaded {split} data, total {len(self.vid_list)}')
    
    def load_annos(self, split, fold_i, ver='split1'):
        df = pd.read_csv(os.path.join(self.root, 'data_info.csv'))
        # row_number
        rows_n = len(df['File_ID'])
        
        # read cross validation set
        vid_list = []
        for i in range(1, 6):
            with open(os.path.join(self.root,'cross_validation', f'{ver}_fold{i}.txt'), 'r') as f:
                vids = f.readlines()
            vids = [v.strip() for v in vids]
            if split=='train':
                if i != fold_i:
                    vid_list += vids
            elif split == 'val':
                if i == fold_i:
                    vid_list += vids            
        
        for i in range(rows_n):
            if df['Split'][i].lower() in ['train', 'val']:
            # vid_list.append(df['File_ID'][i].strip('[]'))
                self.features[df['File_ID'][i].strip('[]')] = {
                    'Adoration': df['Adoration'][i],
                    'Amusement': df['Amusement'][i],
                    'Anxiety': df['Anxiety'][i],
                    'Disgust': df['Disgust'][i],
                    'Empathic-Pain': df['Empathic-Pain'][i],
                    'Fear': df['Fear'][i],
                    'Surprise': df['Surprise'][i],
                    'Age': df['Age'][i],
                    'Country': COUNTRY[df['Country'][i]],
                    'split':df['Split'][i].lower()
                }
        
        for split in ['train', 'val']:
            # load neutral
            npz_file = f'/data/data/ABAW5/challenge4/AffectNet/{split}.npz'
            npz = np.load(npz_file)
            file_path = npz['file_relpath']
            # features = npz['feature']
            pred = npz['expression_pred']
            pred = softmax(pred, axis=1)
            EMOTIONS ={0: 'Neutral', 1: 'Happy', 2: 'Sad', 3: 'Surprise', 4: 'Fear', 5: 'Disgust', 6: 'Anger'}
            n = len(file_path)
            prev_i = 0
            for i in range(n-1):
                if file_path[i][:5] != file_path[i+1][:5]:
                    self.features[file_path[i][:5]]['neutral'] = pred[prev_i:i+1, 0]
                    prev_i = i + 1 
                    
            # load_text_fea
            with open(os.path.join(self.root, f'{split}_debert.pkl'), 'rb') as f:
                data = pickle.load(f)
            for vid, fea in data.items():
                self.features[vid]['deberta'] = fea
        return vid_list
    

    def __getitem__(self, index):
        vid_id = self.vid_list[index]
        data = self.features[vid_id]
        split = data['split']
        label = np.array([data[e] for e in self.EMOTION_NAME])
        sample = {'label': label, 'visual':None, 'audio':None, 'text':None, 
                  'vid':vid_id, 'pretrained_feature':None}

        if self.use_visual:
            frames = np.load(os.path.join(self.root, 'crop_face', split+'_npy', f'{vid_id}.npy'), allow_pickle=True)
            if len(frames) < self.img_max:
                frames = np.concatenate([frames, frames],axis=0)
            n = len(frames)
            img_step = len(frames)/float(self.img_max)
            indexes = np.array([int(i*img_step+random.uniform(0, img_step)) for i in range(self.img_max)])
            frames = frames[indexes]
            if self.shuffle_frame and np.random.randn() < 0.5:
                np.random.shuffle(frames)
            images = []
            for im in frames:
                images.append(Image.fromarray(im))
            if self.transforms:
                images =  [self.transforms(im) for im in images]
            sample['visual'] = torch.stack(images)
            sample['pretrained_feature'] = np.ones(1)
            if 'neutral' in data and indexes[-1] < len(data['neutral']):
                sample['attention_map'] = 1-data['neutral'][indexes]
            else:
                sample['attention_map'] = np.ones(self.img_max)
            if self.use_mae_feature:
                sample['visual'] = np.load(os.path.join(self.root, f'{split}_mae', f'{vid_id}.npy'))[:400]


        if self.use_audio:
            audio = np.load(os.path.join(self.root, f'{split}_hubertbase_npy', f'{vid_id}.npy')).squeeze()
            # sample['audio'] = np.array(audio)[:,:800]
            max_len = min(800, len(audio))
            step_n = len(audio)/max_len
            index = [int(i*step_n) for i in range(max_len-1)]
            sample['audio'] = np.array(audio[index])

        if self.use_text:
            if 'deberta' in data:
                text = data['deberta']
            else:
                text = np.zeros((1, 2, 1024))
            sample['text'] = text
        return sample
     
    def __len__(self):
        return len(self.vid_list)


class Dataset_ERI_MAE_test(Dataset):
    def __init__(self, root, split, img_max=100, transforms=None, fold_i=1, use_mae_feature=False,
                 use_visual=False, use_audio=False, use_text=False, use_dual=False, shuffle_frame=False) -> None:
        # root = '/data/data/ABAW5/challenge4/'
        self.root = root
        self.transforms = transforms
        self.features = {}
        self.vid_list = self.load_annos(split)[:]
        self.img_max = img_max
        self.use_visual = use_visual
        self.use_audio = use_audio
        self.use_text = use_text
        self.shuffle_frame = shuffle_frame
        self.use_mae_feature = use_mae_feature
        self.EMOTION_NAME = ['Adoration', 'Amusement', 'Anxiety', 'Disgust', 'Empathic-Pain', 'Fear', 'Surprise', 'Country']
        
        print(f'loaded {split} data, total {len(self.vid_list)}')
    
    def load_annos(self, split):
        df = pd.read_csv(os.path.join(self.root, 'data_info.csv'))
        # row_number
        rows_n = len(df['File_ID'])
        
        # read cross validation set
        
        for i in range(rows_n):
            if df['Split'][i].lower() in [split]:
            # vid_list.append(df['File_ID'][i].strip('[]'))
                self.features[df['File_ID'][i].strip('[]')] = {
                    'split':df['Split'][i].lower()
                }
        vid_list = list(self.features.keys())
        vid_list.sort()
      
        # load_text_fea
        with open(os.path.join(self.root, f'{split}_debert.pkl'), 'rb') as f:
            data = pickle.load(f)
        for vid, fea in data.items():
            self.features[vid]['deberta'] = fea
        return vid_list
    

    def __getitem__(self, index):
        vid_id = self.vid_list[index]
        data = self.features[vid_id]
        split = data['split']
        sample = {'label': 0, 'visual':None, 'audio':None, 'text':None, 
                  'vid':vid_id, 'pretrained_feature':None}

        if self.use_visual:
            frames = np.load(os.path.join(self.root, 'crop_face', split+'_npy', f'{vid_id}.npy'), allow_pickle=True)
            if len(frames) < self.img_max:
                frames = np.concatenate([frames, frames],axis=0)
            n = len(frames)
            img_step = len(frames)/float(self.img_max)
            indexes = np.array([int(i*img_step+random.uniform(0, img_step)) for i in range(self.img_max)])
            frames = frames[indexes]
            if self.shuffle_frame and np.random.randn() < 0.5:
                np.random.shuffle(frames)
            images = []
            for im in frames:
                images.append(Image.fromarray(im))
            if self.transforms:
                images =  [self.transforms(im) for im in images]
            sample['visual'] = torch.stack(images)
            

        if self.use_audio:
            audio = np.load(os.path.join(self.root, f'{split}_hubertbase_npy', f'{vid_id}.npy'))
            sample['audio'] = np.array(audio)
            
        if self.use_text:
            if 'deberta' in data:
                text = data['deberta']
            else:
                text = np.zeros((1, 2, 1024))
            sample['text'] = text
        return sample
     
    def __len__(self):
        return len(self.vid_list)


def collate_fn3(batch):

    if batch[0]['audio'] is not None:
        sort_key = 'audio'
    else:
        sort_key = 'visual'
    batch = sorted(batch, key=lambda x: x[sort_key].squeeze().shape[0], reverse=True)
    labels = torch.stack([torch.FloatTensor(sample['label']) for sample in batch])
    batch_dict = {'lengths':{}, 'visual':None, 'audio':None, 'text':None}

    # get the data out of the batch - use pad sequence util functions from PyTorch to pad things
    if batch[0]['visual'] is not None:
        # visual = pad_sequence([torch.FloatTensor(sample['visual']).squeeze() for sample in batch], batch_first=True)
        batch_dict['visual'] = torch.stack([torch.FloatTensor(sample['visual']).squeeze() for sample in batch])
        batch_dict['lengths']['v'] = torch.LongTensor([sample['visual'].squeeze().shape[0] for sample in batch])

    if batch[0]['audio'] is not None:
        audio = pad_sequence([torch.FloatTensor(sample['audio']).squeeze() for sample in batch], batch_first=True)
        batch_dict['audio'] = audio
        # batch_dict['lengths']['a'] = torch.LongTensor([np.ceil(sample['audio'].squeeze().shape[0]/3200) for sample in batch])
        batch_dict['lengths']['a'] = torch.LongTensor([np.ceil(sample['audio'].squeeze().shape[0]) for sample in batch])

    if batch[0]['text'] is not None:
        text = pad_sequence([torch.FloatTensor(sample['text']).squeeze() for sample in batch], batch_first=True)
        batch_dict['text'] = text
        batch_dict['lengths']['t'] = torch.LongTensor([sample['text'].squeeze().shape[0] for sample in batch])

    batch_dict['vid'] = [sample['vid'] for sample in batch]    
    return batch_dict

def collate_fn2_audio(batch):

    sort_key = 'audio'
    batch = sorted(batch, key=lambda x: x[sort_key].squeeze().shape[0], reverse=True)
    labels = torch.stack([torch.FloatTensor(sample['label']) for sample in batch])
    batch_dict = {'labels': labels, 'lengths':{}, 'audio':None}

    if batch[0]['audio'] is not None:
        audio = pad_sequence([torch.FloatTensor(sample['audio']).squeeze() for sample in batch], batch_first=True)
        batch_dict['audio'] = audio
        # batch_dict['lengths']['a'] = torch.LongTensor([np.ceil(sample['audio'].squeeze().shape[0]/3200) for sample in batch])
        batch_dict['lengths']['a'] = torch.LongTensor([np.ceil(sample['audio'].squeeze().shape[0]) for sample in batch])
    batch_dict['vid'] = [sample['vid'] for sample in batch]
    
    return batch_dict


def collate_fn2(batch):

    if batch[0]['audio'] is not None:
        sort_key = 'audio'
    else:
        sort_key = 'visual'
    batch = sorted(batch, key=lambda x: x[sort_key].squeeze().shape[0], reverse=True)
    labels = torch.stack([torch.FloatTensor(sample['label']) for sample in batch])
    batch_dict = {'labels': labels, 'lengths':{}, 'visual':None, 'audio':None, 'text':None}

    # get the data out of the batch - use pad sequence util functions from PyTorch to pad things
    if batch[0]['visual'] is not None:
        # visual = pad_sequence([torch.FloatTensor(sample['visual']).squeeze() for sample in batch], batch_first=True)
        try:
            batch_dict['visual'] = torch.stack([torch.FloatTensor(sample['visual']).squeeze() for sample in batch])
        except:
            batch_dict['visual'] = pad_sequence([torch.FloatTensor(sample['visual']).squeeze() for sample in batch], batch_first=True)
            
        batch_dict['lengths']['v'] = torch.LongTensor([sample['visual'].squeeze().shape[0] for sample in batch])

    if batch[0]['audio'] is not None:
        audio = pad_sequence([torch.FloatTensor(sample['audio']).squeeze() for sample in batch], batch_first=True)
        batch_dict['audio'] = audio
        # batch_dict['lengths']['a'] = torch.LongTensor([np.ceil(sample['audio'].squeeze().shape[0]/3200) for sample in batch])
        batch_dict['lengths']['a'] = torch.LongTensor([np.ceil(sample['audio'].squeeze().shape[0]) for sample in batch])


    if batch[0]['text'] is not None:
        text = pad_sequence([torch.FloatTensor(sample['text']).squeeze() for sample in batch], batch_first=True)
        batch_dict['text'] = text
        batch_dict['lengths']['t'] = torch.LongTensor([sample['text'].squeeze().shape[0] for sample in batch])

    if batch[0]['text'] is not None:
        text = pad_sequence([torch.FloatTensor(sample['text']).squeeze() for sample in batch], batch_first=True)
        batch_dict['text'] = text
        # batch_dict['lengths']['a'] = torch.LongTensor([np.ceil(sample['audio'].squeeze().shape[0]/3200) for sample in batch])
        batch_dict['lengths']['t'] = torch.LongTensor([np.ceil(sample['text'].squeeze().shape[0]) for sample in batch])
   
    batch_dict['attention_map'] = torch.stack([torch.FloatTensor(sample['attention_map']).squeeze() for sample in batch])
    batch_dict['vid'] = [sample['vid'] for sample in batch]
    
    return batch_dict
    
def collate_fn(batch):
    '''
    Collate functions assume batch = [Dataset[i] for i in index_set]
    '''
    
    # for later use we sort the batch in descending order of length
    batch = sorted(batch, key=lambda x: x['imgs'].shape[0], reverse=True)
    
    # get the data out of the batch - use pad sequence util functions from PyTorch to pad things

    sentences = pad_sequence([torch.FloatTensor(sample['text']).squeeze() for sample in batch], batch_first=True)
    visual = pad_sequence([torch.FloatTensor(sample['imgs']).squeeze() for sample in batch], batch_first=True)
    acoustic = pad_sequence([torch.FloatTensor(sample['audio']).squeeze() for sample in batch], batch_first=True)
    labels = torch.stack([torch.FloatTensor(sample['label']) for sample in batch])
    
    lengths_t = torch.LongTensor([sample['text'].squeeze().shape[0] for sample in batch])
    lengths_v = torch.LongTensor([sample['imgs'].squeeze().shape[0] for sample in batch])
    lengths_a = torch.LongTensor([sample['audio'].squeeze().shape[0] for sample in batch])
    lengths = {}
    lengths['a'] = lengths_a
    lengths['v'] = lengths_v
    lengths['t'] = lengths_t

    return {'imgs': visual,
                'labels': labels,
                'audio': acoustic,
                'text': sentences,
                'lengths':lengths,
                }
    
# sentences, visual, acoustic, labels, lengths, bert_sentences, bert_sentence_types, bert_sentence_att_mask


if __name__ == '__main__':
    import torchvision.transforms as transforms
    root = '/data/data/ABAW5/challenge4/'
    # model = torch.hub.load('cfzd/FcaNet', 'fca152' ,pretrained=True)
    # model=torch.nn.Sequential(*(list(model.children())[:-1]))
    # print(model)

    # inputs = torch.randn((2,3,224,224))
    # out = model(inputs).view(-1, 2048)
    # print(out.shape)
    # exit()
    # with open(os.path.join(root, 'train'+'_text.txt'),'r') as f:
    #     data = f.readlines()
    # lines = []
    # for line in data:
    #     lines.append(line.replace(';', ' '))
    # with open(os.path.join(root, 'train'+'_text_2.txt'), 'w') as f:
    #     f.writelines(lines)       
    
    transform1 = transforms.Compose([
        transforms.Resize([256,256]),
        transforms.RandomResizedCrop([224, 224], ratio=[0.8,1.2]),
        # transforms.RandomHorizontalFlip(0.5),
        # transforms.RandomRotation([-10,10]),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2,hue=0.15),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

    ])
             
    # dataset_ERI = Dataset_ERI(root, 'val', img_max=4, transforms=transform1)
    
    # dataset_ERI1 = Dataset_ERI_MAE(root, 'train', img_max=16, transforms=transform1, use_visual=True, use_audio=True, use_text=True)
    dataset_ERI2 = Dataset_ERI_MAE(root, 'val', img_max=16, transforms=transform1, use_visual=True, use_audio=True, use_text=True,
                                   use_mae_feature=False)
    # pcc = (dataset_ERI1.pcc * len(dataset_ERI1.vid_list) + dataset_ERI2.pcc*len(dataset_ERI2.vid_list)) / (len(dataset_ERI2.vid_list)+len(dataset_ERI1.vid_list))
    data_loader = DataLoader(
    dataset=dataset_ERI2,
    batch_size=4,
    shuffle=False,
    num_workers=2,
    collate_fn=collate_fn2
    ) 

    for i, inputs in tqdm(enumerate(data_loader), total=len(data_loader)):
        pass