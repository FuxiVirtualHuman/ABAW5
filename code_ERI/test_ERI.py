import os
import pickle
import torch
from models.pipeline5 import Pipeline
from models.single_exp_detect import Single_exp_detect_trans, Single_exp_detect_MISA, Single_exp_detect_Eff
from torch.nn import MSELoss, CrossEntropyLoss, L1Loss, SmoothL1Loss
import numpy as np
import torch.nn.functional as F
from data.dataset_chanllege4 import Dataset_ERI_MAE, collate_fn2, Dataset_ERI_MAE_5split
import torchvision.transforms.transforms as transforms
from torch.utils import data
from torch.utils.data import DataLoader
from torch import optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch import nn
from torch.autograd import Variable
import random
import pandas as pd
from models.EffMulti_modality import EffMulti_modality2
from scipy.stats import pearsonr
import math
from models.linear_modal import Vallina_fusion_visual, mixup_data, Vallina_fusion_visual_all
from sklearn.metrics import accuracy_score, confusion_matrix
import PIL
# from lion_pytorch import Lion
from torch.distributions import MultivariateNormal as MVN
from torch.nn.modules.loss import _Loss
from torch.nn import TripletMarginLoss
from collections import OrderedDict

def test_ERI(epoch, loader, net, best_acc, best_pcc, patience_cur, save_step=10, patience=10, save_res=False):
    # print("train {} epoch".format(epoch))
    net = net.eval()
    b = '{l_bar}{bar:50}{r_bar}{bar:-60b}'
    pbar = tqdm(enumerate(loader), total=len(loader), bar_format=b, ncols=160)
    preds, gt = [], []
    # for results
    vid_id = []
    logger = []
    for i, data in pbar:
        # if i> 10:
        #     preds = np.concatenate(preds)
        #     gt = np.concatenate(gt)
        #     preds_rounds.append(preds)
        #     gt_rounds.append(gt)
        #     return
        vid_id += data['vid']
        vids.append(vid_id)
        continue
        imgs, audios,  labels, lengths,  = data['visual'], data['audio'], data['labels'], data['lengths']
        imgs, labels  = imgs.cuda(), labels.cuda()
        attention_map = data['attention_map'].cuda()

        if use_audio:
            audios = audios.cuda()
            audios2 = data['audio2'].cuda()
            audios3 = data['audio3'].cuda()
        if use_text:
            text = data['text'].cuda()
        else:
            text = None
        pretrained_feature = data['pretrained_feature'].cuda()

        # print(imgs)
        with torch.no_grad():
            pred, fea, labels_mixup = net(imgs, audios, lengths,pretrained_feature, attention_map=attention_map,
                                          return_feature=use_triplets, use_shift=use_shift,
                                          inputs_a2=audios2, inputs_a3=audios3, inputs_t=text)
        vids.append(vid_id)
        preds.append(pred.detach().cpu().numpy())
        gt.append(labels.cpu().numpy())
          
    # logger.append(f'[Test epoch {epoch}]\t pcc:{pcc_avg:.4f}({pcc})\n')
    # with open(os.path.join(log_save_path, f'{timestamp}.log'), "a+") as log_file:
    #     log_file.writelines(logger)
    with open('/data/Workspace/ABAW/code_ABAW5/test/val_vid_list.pkl', 'wb') as f:
        pickle.dump({'vid':vids},f)
    preds = np.concatenate(preds)
    gt = np.concatenate(gt)
    preds_rounds.append(preds)
    gt_rounds.append(gt)

    return

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
     
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Running ERI Training.')
    parser.add_argument('--fold', required=False,type=int, default=0,
                        help='specify fold for validation, 0 for offical, 1-5 for cross validation')
    parser.add_argument('-seed', required=False,type=int, default=20,
                        help='random seed.')
    
    args = parser.parse_args()
    
    import time
    os.chdir('/data/Workspace/ABAW/code_ABAW5/')
    use_cuda = True
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    # pretrained = '20230220-124201_10'
    # models = ['20230312-045712_best'] # use patch
    
    # models = ['20230315-024231_best','20230315-025039_best','20230315-025237_best','20230315-033235_best','20230313-142241_best'] # official fold 

    # models = ['20230310-024649_best', '20230310-024812_best','20230310-024946_best','20230310-090807_best','20230310-091424_best'] # 5fold exp3
    # models = ['20230311-054935_best', '20230311-054950_best','20230311-055021_best','20230313-021353_best','20230311-055204_best'] # 5fold exp3
    models = ['20230313-144258_best','20230313-144719_best','20230314-031629_best','20230314-031925_best','20230314-032030_best'] # 5fold exp4
    # models = ['20230313-063632_best','20230313-063607_best','20230313-063404_best','20230313-063343_best','20230313-063310_best'] # 5fold exp5 
    # models = ['20230313-063226_best','20230314-025820_best','20230313-063142_best','20230313-083941_best','20230313-063035_best'] # 5fold exp6
     
    for i in range(1,6):
        args.fold = 0
        pretrained = models[i-1]
    
    # for pretrained in models:
    
    # pretrained = '20230311-054935_best'

        task = 'ERI' 
        model_name = 'EffMulti'  # ['baseline','mutual','dropout','resnet']
        ck_save_path = f'/data/Workspace/ABAW/code_ABAW5/checkpoints'
        log_save_path = f'/data/Workspace/ABAW/code_ABAW5/logs'
        pred_save_path = f'/data/Workspace/ABAW/code_ABAW5/test'
        os.makedirs(ck_save_path,exist_ok=True)
        os.makedirs(log_save_path, exist_ok=True)
        os.makedirs(pred_save_path, exist_ok=True)

        print(f'************** NOW IS {task} TASK. TIMESTAMP {timestamp} ******************')
        # training parameters
        bz = 16
        lr = 1e-5 # 5e-5
        patience = 10
        save_step = 5
        img_max = 32  # 32
        out_dim = 7  
        use_audio = True
        use_visual = True
        use_text = True
        use_attention = False
        use_triplets = False
        save_res = True
        use_dual = False
        use_ccc = False
        use_shift = False
        use_mixup = False
        use_mea = False
        use_distill = False
        weight_ce = 0.0
        backbone_visual = 'mae' # mae or emb
        fold_i = args.fold
        # loss

        # data
        mean = [0.49895147219604985, 0.4104390648367995, 0.3656147590417074]
        std = [0.2970847084907291, 0.2699003075660314, 0.2652599579468044]

        transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
        
        data_root = '/data/data/ABAW5/challenge4/'
        if fold_i == 0:
            dataset_ERI = Dataset_ERI_MAE
        else:
            dataset_ERI = Dataset_ERI_MAE_5split

        val_dataset = dataset_ERI(data_root, 'val', img_max=img_max, transforms=transform,use_text=use_text, fold_i=fold_i,
                                    use_audio=use_audio, use_visual=use_visual)
        val_dataloader = DataLoader(dataset=val_dataset,batch_size=bz,shuffle=False,num_workers=10,
                                    collate_fn=collate_fn2, drop_last=False, pin_memory=True)
        
        # model
        net = Vallina_fusion_visual(in_dim=768, out_dim=out_dim, backbone_visual=backbone_visual, use_audio=use_audio, use_text=use_text,
                                    use_attention=use_attention, use_dual=use_dual, last_hidden_ratio=2)
        
        net = net.cuda()
        torch.backends.cudnn.benchmark = True
        
        
        
        # average_weights = OrderedDict()
        # for key in net.state_dict():
        #     average_weights[key] = torch.zeros_like(net.state_dict()[key], dtype=float)            
        # for pretrained in [
        #             '20230311-062633', 
        #             '20230310-021658',
        #             '20230315-033235',
        #             '20230315-025237',
        #             '20230315-025039',
        #             '20230315-024231',
        #             '20230314-123450',
        #             # '20230312-045712',
        #             ] :
        #     checkpoint = torch.load(os.path.join(ck_save_path, f'{pretrained}_best.pt'))
        #     for key in checkpoint['state_dict']:
        #         average_weights[key] += checkpoint['state_dict'][key]
        # for key in average_weights:
        #    average_weights[key] /= 7.
        # net.load_state_dict(average_weights)

        
        checkpoint = torch.load(os.path.join(ck_save_path, f'{pretrained}.pt'))
        
        try:
            net.load_state_dict(checkpoint['state_dict'])
        except:
            net.load_state_dict({key.replace('module.',''):checkpoint['state_dict'][key] for key in checkpoint['state_dict']
                if 'repvgg' not in key and 'relation' not in key })
        print("=> loaded pretrained model {}".format(pretrained))
        scaler = GradScaler()
        
        patience_cur = patience
        step_i = 0
        best_score = float('inf')
        best_pcc = 0

        preds_rounds, gt_rounds, vids = [], [], []
        start_n = 0
        pred_filename = os.path.join(pred_save_path, f'{pretrained}_val.pkl')
        # if os.path.exists(pred_filename):
        #     with open(pred_filename, 'rb') as f:
        #         data = pickle.load(f)
        #     preds_rounds, gt_rounds = data['preds'], data['gt']
        #     start_n = len(data['preds'])    

        for epoch in range(start_n, 5):
            test_ERI(epoch, val_dataloader, net, best_score, best_pcc, patience_cur, 
                                                    save_step=save_step, patience=patience, save_res=save_res)
            # with open(pred_filename, 'wb') as f:
            #     pickle.dump({'preds':preds_rounds, 'gt':gt_rounds, 'vid':vids},f)

                