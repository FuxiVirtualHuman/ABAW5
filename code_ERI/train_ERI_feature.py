import os
import pickle
import torch
from models.pipeline5 import Pipeline
from models.single_exp_detect import Single_exp_detect_trans, Single_exp_detect_MISA, Single_exp_detect_Eff
from torch.nn import MSELoss, CrossEntropyLoss, L1Loss, SmoothL1Loss
import numpy as np
import torch.nn.functional as F
from data.dataset_chanllege4 import Dataset_ERI_MAE, collate_fn2
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
from models.linear_modal import Vallina_fusion

def concordance_correlation_coefficient(y_true, y_pred,
                                        sample_weight=None,
                                        multioutput='uniform_average'):
    """Concordance correlation coefficient.
    The concordance correlation coefficient is a measure of inter-rater agreement.
    It measures the deviation of the relationship between predicted and true values
    from the 45 degree angle.
    Read more: https://en.wikipedia.org/wiki/Concordance_correlation_coefficient
    Original paper: Lawrence, I., and Kuei Lin. "A concordance correlation coefficient to evaluate reproducibility." Biometrics (1989): 255-268.
    Parameters
    ----------
    y_true : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Estimated target values.
    Returns
    -------
    loss : A float in the range [-1,1]. A value of 1 indicates perfect agreement
    between the true and the predicted values.
    Examples
    # --------
    # >>> from sklearn.metrics import concordance_correlation_coefficient
    # >>> y_true = [3, -0.5, 2, 7]
    # >>> y_pred = [2.5, 0.0, 2, 8]
    # >>> concordance_correlation_coefficient(y_true, y_pred)
    # 0.97678916827853024
    # """
    cor = np.corrcoef(y_true, y_pred)[0][1]

    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)

    var_true = np.var(y_true)
    var_pred = np.var(y_pred)

    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)

    numerator = 2 * cor * sd_true * sd_pred

    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2

    return numerator / denominator


def CCC_loss(x, y):
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    rho = torch.sum(vx * vy) / ((torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2))) + 1e-10)
    x_m = torch.mean(x)
    y_m = torch.mean(y)
    x_s = torch.std(x)
    y_s = torch.std(y)
    ccc = 2 * rho * x_s * y_s / ((x_s ** 2 + y_s ** 2 + (x_m - y_m) ** 2) + 1e-10)
    return 1 - ccc


class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classi?ed examples (p > .5),
                                   putting more focus on hard, misclassi?ed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
            for i in range(class_num):
                self.alpha[i, :] = 0.25
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs, dim=1)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        # print(class_mask)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()
        # print('probs size= {}'.format(probs.size()))
        # print(probs)

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        # print('-----bacth_loss------')
        # print(batch_loss)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


def train_ERI(epoch, loader, net, optimizer, best_Exp_score, net3=None):
    print(f"* Epoch {epoch}, lr:{lr:.5f}, timestamp:{timestamp}")
    preds, gt = [], []
    loss_sum = 0.0
    net = net.train()
    # print('Total batch',len(loader))
    step_n = len(loader)
    b = '{l_bar}{bar:40}{r_bar}{bar:-10b}'
    pbar = tqdm(enumerate(loader), total=len(loader), bar_format=b, ncols=160)
    logger = []
    time0 = time.time()
    for i, data in pbar:
        imgs, labels, lengths = data['audio'], data['labels'], data['lengths']
        imgs, labels = imgs.cuda(), labels.cuda()
        time_data = time.time() - time0
        optimizer.zero_grad()

        pred = net(imgs, lengths)
        # Exp_loss

        # loss_l1 = criterion_l1(pred, labels)
        loss_l1 = criterion_l2(pred, labels)
        loss = loss_l1

        loss.backward()
        optimizer.step()
        # scheduler.step(i)
        lr = optimizer.param_groups[0]['lr']

        loss_sum += loss.item()
        avg_loss = loss_sum / (i+1)
        preds.append(pred.detach().cpu().numpy())
        gt.append(labels.cpu().numpy())
        time_train = time.time() - time0 - time_data  
        time0 = time.time()      
        pbar.set_description(f'[Train epoch {epoch}]\t loss:{loss.item():.4f}({avg_loss:.4f}) time_data:{time_data:.1f} time_train:{time_train:.1f}' )
        # pbar.set_description(f'[Train epoch {epoch}(lr:{lr:.5f})]\t loss:{loss.item():.4f}({avg_loss:.4f})' )
        logger.append(f"epoch: {epoch}, step: {i},  Loss: {loss.item():.4f}\n")
    
    scheduler.step(epoch)
    # metrics (pcc)
    preds = np.concatenate(preds)
    gt = np.concatenate(gt)
    pcc = [np.round(pearsonr(preds[:,i], gt[:,i])[0], 4) for i in range(7)]
    pcc_avg = np.mean(np.array(pcc))
    
    print(f'[Train epoch {epoch}]\t loss_avg: {avg_loss:.4f}\t pcc:{pcc_avg:.4f}({pcc})')
    logger.append(f'[Train epoch {epoch}]\t loss_avg: {avg_loss:.4f}\t pcc:{pcc_avg:.4f}({pcc})\n')
    with open(os.path.join(log_save_path, f'{timestamp}.log'), "a+") as log_file:
        log_file.writelines(logger)

    return avg_loss


def test_ERI(epoch, loader, net, best_acc, patience_cur, save_step=10, patience=10):
    # print("train {} epoch".format(epoch))
    preds, gt = [], []
    net = net.eval()
    loss_sum = 0
    b = '{l_bar}{bar:50}{r_bar}{bar:-60b}'
    pbar = tqdm(enumerate(loader), total=len(loader), bar_format=b, ncols=160)
    
    logger = []
    for i, data in pbar:
        imgs, labels, lengths = data['audio'], data['labels'], data['lengths']
        
        imgs, labels = imgs.cuda(), labels.cuda()
        # print(imgs)
        
        pred = net(imgs, lengths)
        # Exp_loss
        # loss_l1 = criterion_l1(pred, labels)
        loss_l1 = criterion_l2(pred, labels)

        loss = loss_l1
        loss_sum += loss.item()
        avg_loss = loss_sum / (i+1)

        preds.append(pred.detach().cpu().numpy())
        gt.append(labels.cpu().numpy())

        pbar.set_description(f'[Test epoch {epoch}]\t loss:{loss.item():.4f}({avg_loss:.4f})')
          
    # metrics (pcc)
    preds = np.concatenate(preds)
    gt = np.concatenate(gt)
    pcc = [np.round(pearsonr(preds[:,i], gt[:,i])[0], 4) for i in range(7)]
    pcc_avg = np.mean(np.array(pcc))
      
    print(f'[Test epoch {epoch}]\t loss_avg: {avg_loss:.4f}\t pcc:{pcc_avg:.4f}({pcc})\n')
    logger.append(f'[Test epoch {epoch}]\t loss_avg: {avg_loss:.4f}\t pcc:{pcc_avg:.4f}({pcc})\n')
    with open(os.path.join(log_save_path, f'{timestamp}.log'), "a+") as log_file:
        log_file.writelines(logger)
    

    
    metric = avg_loss
    if epoch % save_step == 0:
        torch.save({'state_dict': net.state_dict()}, os.path.join(ck_save_path,f'{timestamp}_{epoch}.pt'))
    if metric < best_acc:
        patience_cur = patience
        best_acc = metric
        torch.save({'state_dict': net.state_dict()}, os.path.join(ck_save_path,f'{timestamp}_best.pt'))
        print("Found new best model, saving to disk...")
    else:
        patience_cur -= 1    
    
    return best_acc


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
     
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


if __name__ == '__main__':
    import time
    setup_seed(20)
    use_cuda = True
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    pretrained = ''
    
    task = 'ERI' 
    model_name = 'EffMulti'  # ['baseline','mutual','dropout','resnet']
    ck_save_path = f'./checkpoints'
    log_save_path = f'./logs'
    pred_save_path = f'./test'
    os.makedirs(ck_save_path,exist_ok=True)
    os.makedirs(log_save_path, exist_ok=True)
    os.makedirs(pred_save_path, exist_ok=True)

    print(f'************** NOW IS {task} TASK. TIMESTAMP {timestamp} ******************')
    # training parameters
    bz = 16
    lr = 0.0005 
    patience = 10
    save_step = 5
    img_max = 32
    use_audio = True
    use_visual = False
    
    # LOSS
    criterion = L1Loss()

    # data
    transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor()])

    data_root = '/data/data/ABAW5/challenge4/'
    train_dataset = Dataset_ERI_MAE(data_root, 'train', img_max=img_max, transforms=transform,
                                    use_audio=use_audio, use_visual=use_visual)
    train_dataloader = DataLoader(dataset=train_dataset,batch_size=bz,shuffle=True,num_workers=20,collate_fn=collate_fn2, drop_last=True)
    
     
    val_dataset = Dataset_ERI_MAE(data_root, 'val', img_max=img_max, transforms=transform,
                                  use_audio=use_audio, use_visual=use_visual)
    val_dataloader = DataLoader(dataset=val_dataset,batch_size=bz,shuffle=False,num_workers=20,collate_fn=collate_fn2, drop_last=True)
    
    
    # model
    net = Vallina_fusion(in_dim=768, out_dim=7)
    # net = nn.DataParallel(net)
    net = net.cuda()
    
    if pretrained:
        checkpoint = torch.load(os.path.join(ck_save_path, f'{pretrained}.pt'))
        net.load_state_dict(checkpoint['state_dict'])
        print("=> loaded pretrained model {}".format(pretrained))
    else:
        net.apply(init_weights)

    # loss
    criterion_l1 = nn.L1Loss(reduction='mean')
    criterion_l2 = nn.MSELoss(reduction='mean')

    # optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
    #                       weight_decay=1e-4)
    # scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 5, 2, 1e-6)
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, betas=(0.0, 0.99))
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 5, 2, 1e-7)
    
    scaler = GradScaler()
    
    patience_cur = patience
    step_i = 0
    best_pcc_score = 100
    # os.system("git add .")
    # os.system(f"git commit -m  training_{task}_{timestamp}")
    # os.system("git push")
    # torch.multiprocessing.set_start_method('spawn')
    for epoch in range(500000000):
        avg_loss = train_ERI(epoch, train_dataloader, net, optimizer, best_pcc_score)
        best_pcc_score = test_ERI(epoch, val_dataloader, net, best_pcc_score, patience_cur, 
                                  save_step=save_step, patience=patience)

