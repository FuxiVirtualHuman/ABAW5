import csv
import os.path
import random
import pandas as pd
from collections import Counter
from tqdm import tqdm

random.seed(0)
def split_5fold_and_val(task,fold_n=5, root='../annos'):

    print('Task: ', task)
    root = root
    pd_data1 = pd.read_csv(os.path.join(root, f'ABAW3_new_{task}_training.csv'))
    pd_data2 = pd.read_csv(os.path.join(root, f'ABAW3_new_{task}_validation.csv'))

    pd_data = pd.concat([pd_data1, pd_data2])
    pd_img = pd_data.to_dict()['img']
    video_counter = Counter([pi.split("/")[0] for pi in pd_img.values()])
    video_names = list(video_counter.keys())
    random.shuffle(video_names)
    video_names_val = video_names[:int(len(video_names)*0.1)]
    video_names_fold = video_names[int(len(video_names)*0.1):]
    n_lenth = len(video_names_fold) / 5.

    pd_data_test = pd.concat([pd_data[pd_data['img'].str.contains(vname)] for vname in video_names_val])
    pd_data_test.to_csv(os.path.join(root,f'random_{task}_test.csv'), index=False)
    print('Test set saved!')
    print('Processing 5 fold training/val set ....')
    for i in tqdm(range(fold_n)):
        names_train = video_names_fold[int(i*n_lenth):int((i+1)*n_lenth)]
        names_test = video_names_fold[:int(i*n_lenth)] + video_names_fold[int((i+1)*n_lenth):]
        pd_data_train = pd.concat([pd_data[pd_data['img'].str.contains(vname)] for vname in names_train])
        pd_data_val = pd.concat([pd_data[pd_data['img'].str.contains(vname)] for vname in names_test])
        pd_data_train.to_csv(os.path.join(root, f'random_{task}_train_{i}.csv'), index=False)
        pd_data_val.to_csv(os.path.join(root, f'random_{task}_validation_{i}.csv'), index=False)

    print('done!')
    return

def split_5fold_MRI(fold_n=5):
    root = '/data/data/ABAW5/challenge4'
    save_root = os.path.join(root, 'cross_validation')
    df = pd.read_csv(os.path.join(root, 'data_info.csv'))
    os.makedirs(save_root, exist_ok=True)
    rows_n = len(df['File_ID'])
    vid_list = [df['File_ID'][i].strip('[]') for i in range(rows_n) if df['Split'][i].lower() in ['train', 'val']]
    total = len(vid_list)
    random.shuffle(vid_list)
    
    split_n = total/fold_n
    for i in range(fold_n):
        foldi = vid_list[int(split_n*i):int(split_n*(i+1))]
        foldi = [line+'\n' for line in foldi]
        print(f'fold{i}:{len(foldi)}')
        with open(os.path.join(save_root, f'split1_fold{i+1}.txt'), 'w') as f:
            f.writelines(foldi)        
    
    
    

if __name__ == '__main__':
    # split_5fold_and_val('AU', fold_n=5, root='../annos')
    split_5fold_MRI()
