import os
import numpy as np
import pickle
from scipy.stats import pearsonr
from copy import deepcopy


vid_happies=['11111',
'11147',
'11170',
'11183',
'11199',
'11250',
'11274',
'11320',
'11389',
'11397',
'11425',
'11447',
'11495',
'11633',
'11635',
'11650',
'11703',
'11759',
'11765',
'11796',
'11862',
'11894',
'11930',
'11970',
]

def post_process_for_one(timestamp):
    pred_save_path = f'/data/Workspace/ABAW/code_ABAW5/test'
    with open('/data/Workspace/ABAW/code_ABAW5/test/val_vid_list.pkl', 'rb') as f:
        vid_list = pickle.load(f)
    vid_list = vid_list['vid'][0]
    
    filename = os.path.join(pred_save_path, f'{timestamp}_best_val.pkl')
    if not os.path.exists(filename):
        filename = os.path.join(pred_save_path, f'{timestamp}_val.pkl')

    with open(filename, 'rb') as f:
        data = pickle.load(f)
    gt, pred = np.array(data['gt'])[:,:,:7], np.array(data['preds'])[:,:,:7]
    round_num = len(gt)
    pccs = []
    pccs_avg = []
    for n in range(round_num):
        pcci = [np.round(pearsonr(pred[n][:,i], gt[n][:,i])[0], 4) for i in range(7)]
        pcci_avg = np.mean(pcci)
        pccs.append(pcci)
        pccs_avg.append(pcci_avg)
        print(pcci_avg, pcci)
    print(np.mean(pccs_avg))
    
    process1 = 'avg'
    print('='*30, process1, '='*30)
    pred_avg = np.mean(pred, axis=0) # 多次测试投票结果
    pcc = [np.round(pearsonr(pred_avg[:,i], gt[0][:,i])[0], 4) for i in range(7)]
    pcc_avg = np.mean(pcc)
    print(pcc_avg, pcc)

    # pcc = []
    # for i in range(7):
    #     pcci = []
    #     for j in range(7):
    #         pcci.append(pearsonr(pred_avg[i],pred_avg[j])[0])
    #     pcc.append(pcci)
    #     print(pcci)
    # print(pcc)
    
    pred_best = []
    for e in range(7):
        pcc_emotion = [np.round(pearsonr(pred[i,:,e], gt[0][:,e])[0], 4) for i in range(5)]
        best_idx = np.argmax(pcc_emotion)
        pred_best.append(pred[best_idx,:,e])
    pred_best = np.stack(pred_best, axis=1)
    pcc = [np.round(pearsonr(pred_best[:,i], gt[0][:,i])[0], 4) for i in range(7)]
    pcc_avg = np.mean(pcc)
    print(pcc_avg, pcc)
    
    # 把最大值设为1
    process2 = 'argmax'
    pred2 = deepcopy(pred_avg)
    print('='*30, process2, '='*30)
    
    for i in range(7):
        mi,ma = min(pred_avg[:, i]),max(pred_avg[:, i])

        pred2[:,i] *= 1/ma
        # pred2[:,5] *= 1/ma
    pred2 = np.clip(pred2, 0, 1)

    happy_index = [vid_list.index(hi) for hi in vid_happies]
    pred2[happy_index,1] =1 
    # for i in range(len( pred2)):
    #     if pred2[i, 1] > 0.95:
    #         pred2[i,1] = 1
    #     if pred2[i, 5] < 0.3:
    #         pred2[i,5] = 0
        # if pred_avg[i, 5] < 0.08:
            # pred2[i,5] = 0
        # pred2[i,5] += 0.05
    preds_class = np.argmax(pred2[:,:7], axis=1)
    for i in range(len(preds_class)):
        pred2[i, preds_class[i]] = 1 
    pcc = [np.round(pearsonr(pred2[:,i], gt[0][:,i])[0], 4) for i in range(7)]
    pcc_avg = np.mean(pcc)
    print(pcc_avg, pcc)
    
    # clip
    process3 = 'clip'
    pred3 = deepcopy(pred_avg)
    print('='*30, process3, '='*30)
    pred3 = np.clip(pred3, 0, 1)
    pcc = [np.round(pearsonr(pred3[:,i], gt[0][:,i])[0], 4) for i in range(7)]
    pcc_avg = np.mean(pcc)
    print(pcc_avg, pcc)
    
    return pred_avg, gt[0]
    
    # pcc = [np.round(pearsonr(pred[:,i], gt[:,i])[0], 4) for i in range(7)]
    # pcc_avg = np.mean(pcc)
    # print('pcc:', pcc)
    # print('avg:', pcc_avg)
    
def model_ensemble(timestamps):
    preds, gt =[], []
    for timestamp in timestamps:
        print('='*60)

        predi, gti = post_process_for_one(timestamp)
        preds.append(predi)
        gt.append(gti)
    
    
    preds = np.array(preds)
    pred_best = []
    for e in range(7):
        pcc_emotion = [np.round(pearsonr(preds[i,:,e], gt[0][:,e])[0], 4) for i in range(len(preds))]
        best_idx = np.argmax(pcc_emotion)
        print(best_idx)
        pred_best.append(preds[best_idx,:,e])
    pred_best = np.stack(pred_best, axis=1)
    pcc = [np.round(pearsonr(pred_best[:,i], gt[0][:,i])[0], 4) for i in range(7)]
    pcc_avg = np.mean(pcc)
    print(pcc_avg, pcc)
    
    
    
    preds_avg = np.mean(preds, 0)
    pcc = [np.round(pearsonr(preds_avg[:,i], gt[0][:,i])[0], 4) for i in range(7)]
    pcc_avg = np.mean(pcc)
    print('='*15,'model ensemble:', timestamps, '='*15)
    print(pcc_avg, pcc)
    
    
    preds2 = np.mean([preds_avg, pred_best], axis=0)
    pcc = [np.round(pearsonr(preds2[:,i], gt[0][:,i])[0], 4) for i in range(7)]
    pcc_avg = np.mean(pcc)
    print('='*15,'model ensemble:', timestamps, '='*15)
    print(pcc_avg, pcc)
    
if __name__ == '__main__':
    
    # post_process_for_one('ensemble')
    # post_process_for_one('20230311-054935')
    # fold1 = ['20230310-024649',
    #                 '20230311-054935', 
    #                 '20230313-144258',
    #                 '20230313-063632',
    #                 '20230313-063226',
    #                 ]
    fold0 = ['20230312-045712','20230315-024231','20230314-123450',]
    
    # fold2 = ['20230310-024812', '20230311-054950', '20230313-144719', '20230313-063607', '20230314-025820']
    # fold3 = ['20230310-024946', '20230311-055021', '20230314-031629', '20230313-063404', '20230313-063142']
    # fold4 = ['20230310-090807', '20230313-021353', '20230314-031925', '20230313-063343', '20230313-083941']
    # fold5 = ['20230310-091424', '20230311-055204', '20230314-032030', '20230313-063310', '20230313-063035']
    
    model_ensemble(fold0)
