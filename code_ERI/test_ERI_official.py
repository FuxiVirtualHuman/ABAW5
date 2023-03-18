import os
import pickle
import torch
import numpy as np
from data.dataset_chanllege4 import collate_fn3,Dataset_ERI_MAE_test
import torchvision.transforms.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import random
from scipy.stats import pearsonr
from models.linear_modal import Vallina_fusion_visual


def test_ERI(epoch, loader, net, best_acc, best_pcc, patience_cur, save_step=10, patience=10, save_res=False):
    # print("train {} epoch".format(epoch))
    net = net.eval()
    b = '{l_bar}{bar:50}{r_bar}{bar:-60b}'
    pbar = tqdm(enumerate(loader), total=len(loader), bar_format=b, ncols=160)
    preds = []
    # for results
    vid_id = []
    for i, data in pbar:
        imgs, audios, lengths,  = data['visual'], data['audio'], data['lengths']
        imgs = imgs.cuda()
        
        if use_audio:
            audios = audios.cuda()
        if use_text:
            text = data['text'].cuda()
        
        with torch.no_grad():
            pred, _, _ = net(imgs, audios, lengths, inputs_t=text)
        vid_id += data['vid']
        preds.append(pred.detach().cpu().numpy())

    preds = np.concatenate(preds)
    preds_rounds.append(preds)
    vids.append(vid_id)
    return

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':
    import time
    os.chdir('/data/Workspace/ABAW/code_ABAW5/')
    use_cuda = True
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    # pretrained = '20230220-124201_10'

    task = 'ERI' 
    model_name = 'EffMulti'  # ['baseline','mutual','dropout','resnet']
    ck_save_path = f'/data/Workspace/ABAW/code_ABAW5/checkpoints'
    pred_save_path = f'/data/Workspace/ABAW/code_ABAW5/test'

    print(f'************** NOW IS {task} TASK.******************')
    # training parameters
    bz = 16
    lr = 1e-5 # 5e-5
    patience = 10
    save_step = 5
    img_max = 32  # 32
    out_dim = 14
    use_audio = True
    use_visual = True
    use_text = True
    save_res = True
    weight_ce = 0.0
    # pretrained = '20230312-045712_best'
    
    backbone_visual = 'mae' # mae or emb
    # loss

    # data
    mean = [0.49895147219604985, 0.4104390648367995, 0.3656147590417074]
    std = [0.2970847084907291, 0.2699003075660314, 0.2652599579468044]

    transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])
    
    data_root = '/data/data/ABAW5/challenge4/'
    val_dataset = Dataset_ERI_MAE_test(data_root, 'test', img_max=img_max, transforms=transform,use_text=use_text,
                                  use_audio=use_audio, use_visual=use_visual)
    val_dataloader = DataLoader(dataset=val_dataset,batch_size=bz,shuffle=False,num_workers=12,
                                collate_fn=collate_fn3, drop_last=False, pin_memory=True)
    
    # model
    net = Vallina_fusion_visual(in_dim=768, out_dim=out_dim, backbone_visual=backbone_visual, use_audio=use_audio,
                                use_text=use_text, last_hidden_ratio=0.5)
    net = net.cuda()
    torch.backends.cudnn.benchmark = True    
    scaler = GradScaler()
    
    patience_cur = patience
    step_i = 0
    best_score = float('inf')
    best_pcc = 0
    preds_rounds, vids = [], []
    start_n = 0
    
    
    # timestamps = ['20230313-063632_best','20230313-063607_best','20230313-063404_best','20230313-063343_best','20230313-063310_best'] \
        # + ['20230313-063226_best','20230314-025820_best','20230313-063142_best','20230313-083941_best','20230313-063035_best']

    # timestamps = ['20230311-054935_best', '20230311-054950_best','20230311-055021_best','20230313-021353_best','20230311-055204_best'] \
    #     + ['20230313-144258_best','20230313-144719_best','20230314-031629_best','20230314-031925_best','20230314-032030_best'] # 5fold exp4

    timestamps = [
                #   '20230310-021658_best','20230311-062633_best',
                #   '20230314-123450_best', 
                #   '20230315-024231_best',
                #   '20230315-025039_best',
                  '20230315-025237_best','20230315-033235_best'
                  ] # official fold 

    for ti, pretrained in enumerate(timestamps):    
        print(f'testing {ti}/{len(timestamps)}')
        checkpoint = torch.load(os.path.join(ck_save_path, f'{pretrained}.pt'))
        try:
            net.load_state_dict(checkpoint['state_dict'])
        except:
            net.load_state_dict({key.replace('module.',''):checkpoint['state_dict'][key] for key in checkpoint['state_dict']
                if 'repvgg' not in key and 'relation' not in key })
        
        # net.load_state_dict(checkpoint['state_dict'])
        print("=> loaded pretrained model {}".format(pretrained))
        
        pred_filename = os.path.join(pred_save_path, f'{pretrained}_test.pkl')
        if os.path.exists(pred_filename):
            with open(pred_filename, 'rb') as f:
                data = pickle.load(f)
            preds_rounds, vids = data['preds'], data['vid']
            start_n = len(data['preds'])    

        for epoch in range(10):
            test_ERI(epoch, val_dataloader, net, best_score, best_pcc, patience_cur, 
                                                    save_step=save_step, patience=patience, save_res=save_res)
            with open(os.path.join(pred_save_path, f'{pretrained}_test.pkl'), 'wb') as f:
                pickle.dump({'preds':preds_rounds, 'vid':vids}, f)
        
        
