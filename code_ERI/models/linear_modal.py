import os
import torch
from torch import nn
import sys
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
# import s3prl.hub as hub
# from transformers import AutoTokenizer, AutoModel, AutoConfig, Wav2Vec2Processor 
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(curr_path, "../"))
import models.models_vit as models_vit
from models.pipeline_student_InceptionResnet import Pipeline_Incep
# from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoderLayer, TransformerDecoder
import math
from models.TimesNet import Model as Timesnet
from models.repVGG import get_RepVGGplus_func_by_name
from models.TCN import TemporalConvNet
import numpy as np
# import insightface
import torchvision
from tqdm import tqdm
import torchvision.transforms.transforms as transforms
from PIL import Image

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    y = lam * y_a + (1 - lam) * y_b
    return mixed_x, y, lam

class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        # self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        # self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        # src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        return torch.mean(output, dim=1)
    
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
class PrivateEncoder(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, enforce_sorted=False):
        super(PrivateEncoder, self).__init__()
        self.enforce_sorted =enforce_sorted
        self.lstm1 = nn.GRU(in_size, in_size, 1, bidirectional=True, batch_first=True)
        self.lstm2 = nn.GRU(in_size*2, in_size, 1, bidirectional=True, batch_first=True)
        self.layernorm = nn.LayerNorm(in_size*2)
        self.linear1 = nn.Linear(in_size*4, out_size)

    def extract_features(self, sequence, lengths, rnn1, rnn2, layer_norm, do_ptflops=False):
        packed_sequence = pack_padded_sequence(sequence, lengths.cpu(), batch_first=True, enforce_sorted=self.enforce_sorted)
        packed_h1, final_h1 = rnn1(packed_sequence)
        padded_h1, _ = pad_packed_sequence(packed_h1, batch_first=True)
        normed_h1 = layer_norm(padded_h1)
        packed_normed_h1 = pack_padded_sequence(normed_h1, lengths.cpu(),enforce_sorted=self.enforce_sorted, batch_first=True)
        _, final_h2 = rnn2(packed_normed_h1)
        return final_h1, final_h2

    def extract_features_ptflops(self, sequence, lengths, rnn1, rnn2, layer_norm):
        paded_h1, final_h1 = rnn1(sequence)
        normed_h1 = layer_norm(paded_h1)
        _, final_h2 = rnn2(normed_h1)
        return final_h1, final_h2

    def forward(self, x, lengths):
        self.lstm1.flatten_parameters()
        self.lstm2.flatten_parameters()
        batch_size = lengths.size(0)
        final_h1, final_h2 = self.extract_features(x, lengths, self.lstm1, self.lstm2, self.layernorm)
        h = torch.cat((final_h1, final_h2), dim=2).permute(1,0,2).contiguous().view(batch_size, -1)
        y = F.leaky_relu(self.linear1(h))
        return y

class LinearBlock(nn.Module):
    def __init__(self,in_dim=512, out_dim=139):
        super(LinearBlock,self).__init__()
        #self.landmark_net = landmark_2_emb(in_channel)
        self.main = nn.Sequential(
            nn.Linear(in_dim,256),
            nn.ReLU(),
            nn.BatchNorm1d(256),

            nn.Linear(256,128),
            nn.ReLU(),
            nn.BatchNorm1d(128),

            nn.Linear(128,256),
            nn.ReLU(),
            nn.BatchNorm1d(256),

            nn.Linear(256,512),
            nn.ReLU(),
            nn.BatchNorm1d(512),

            nn.Linear(512,256),
            nn.ReLU(),
            nn.BatchNorm1d(256),

            nn.Linear(256,200),
            nn.ReLU(),
            nn.BatchNorm1d(200),
            nn.Linear(200,out_dim),
        )

        
    def forward(self,emb):
        rigs = self.main(emb)
        rigs = torch.sigmoid(rigs)
        return rigs


class predict(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dim=16, dropout_prob=0.5):

        super().__init__()
        self.post_fusion_dim = in_dim
        self.output_dim = out_dim
        self.pre_layer_2 = nn.Linear(self.post_fusion_dim, hidden_dim)
        self.pre_layer_4 = nn.Linear(hidden_dim, self.output_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, out):

        out = self.dropout(out)
        out =  F.leaky_relu(self.pre_layer_2(out))
        # out = F.normalize(out, dim=1)

        out = self.pre_layer_4(out)
        return out
    
class Vallina_fusion(nn.Module):
    def __init__(self,in_dim=512, out_dim=139):
        super(Vallina_fusion,self).__init__()
        # mode_name = 'hubert'
        # model_4 = getattr(hub, mode_name)()
        device = 'cuda'  # or cpu
        # self.model_hubert = model_4.to(device).eval()
        
        # self.model_mae = getattr(models_vit, 'vit_base_patch16')(
        #                 global_pool=True,
        #                 num_classes=9,
        #                 drop_path_rate=0.1,
        #                 img_size=224,
        #             )
        # ckpt_mae = '/data/Workspace/ABAW/code_ABAW5/checkpoints/model-20.pth'
        # print(f"Load pre-trained checkpoint from: {ckpt_mae}")
        # checkpoint = torch.load(ckpt_mae, map_location='cpu')
        # checkpoint_model = checkpoint['model']
        # self.model_mae.load_state_dict(checkpoint_model, strict=False)
        # self.model_mae.to(device).train()
        
        # self.model_emb = Pipeline_Incep(dropout_prob=0) #0.6
        # self.visual_encoder = PrivateEncoder(in_size=512, hidden_size=32, out_size=64, enforce_sorted=True)
        self.visual_encoder = PrivateEncoder(in_size=512, hidden_size=32, out_size=64, enforce_sorted=True)
        self.audio_encoder = PrivateEncoder(in_size=768, hidden_size=32, out_size=64, enforce_sorted=True)
        # self.linear_block = LinearBlock(64, out_dim)
        self.linear_block = predict(in_dim=64, out_dim=out_dim, dropout_prob=0.5) #*2

    def forward(self, inputs, length):
        # b, n, c, h, w = inputs.shape
        # images_batch = torch.reshape(inputs, (b*n, c, h, w))
        # self.model_emb.eval()
        # with torch.no_grad():
        #     x = self.model_emb.forward_fea(images_batch)

        # _, x = self.model_mae(images_batch, ret_feature=True)
        # x =  torch.reshape(x, (b, n, -1))
        # x = self.visual_encoder(x, length['v'])

        # with torch.no_grad():
        #     x = self.model_hubert(inputs)['hidden_states'][0]
        x = self.audio_encoder(inputs, length['a'])
        
        # x = torch.mean(inputs, dim=1)
        x = self.linear_block(x)
        out = F.leaky_relu(x)
        
        return out

def shift(x, n_segment=4, fold_div=3, inplace=False):
    # nt, c, h, w or nt, c, d        
    size = x.size()
    nt, c = x.size()[0], x.size()[1]
    n_batch = nt // n_segment
    x = x.view(n_batch, n_segment, c, *size[2:])

    fold = c // fold_div
    out = torch.zeros_like(x)
    out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
    out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
    out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift

    return out.view(size)
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c = x.size()[0], x.size()[1]
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)
    
class Vallina_fusion_visual(nn.Module):
    def __init__(self,in_dim=512, out_dim=139, backbone_visual='emb', use_audio=False, use_attention=False, 
                 use_dual=False, use_text=False, last_hidden_ratio=0.5):
        super(Vallina_fusion_visual,self).__init__()
        # mode_name = 'hubert'
        # model_4 = getattr(hub, mode_name)()
        # self.model_hubert = model_4.to(device).eval()
        device = 'cuda'  # or cpu
        self.backbone_visual = backbone_visual
        self.use_attention = use_attention
        self.use_dual = use_dual
        self.use_audio = use_audio
        self.use_text = use_text
        self.use_multihead = False
        self.use_patch = False
        self.use_cnn = False
        self.use_visual = True
        self.use_mfcc= True
        print('use patch:', self.use_patch)
        hidden_dim = 128 # 32
        
        # self.backbone_emb = Pipeline_Incep(dropout_prob=0.5) #0.6
        # visual_dim = 512
        
        if backbone_visual=='emb':
            self.backbone_emb = Pipeline_Incep(dropout_prob=0.5) #0.6
            visual_dim = 512
            # for p in self.backbone_emb.parameters():
                # p.requires_grad = False
        elif backbone_visual == 'fca':
            # https://github.com/cfzd/FcaNet
            model = torch.hub.load('cfzd/FcaNet', 'fca50', pretrained=True)
            self.backbone_fca = torch.nn.Sequential(*(list(model.children())[:-1]))
            visual_dim = 2048
        else:
            # vit_large_patch16 or vit_base_patch16
            model_name = 'vit_base_patch16'
            train_data = 'affectnet'
            if model_name== 'vit_base_patch16':
                visual_dim = 768
            else:
                visual_dim = 1024
            if train_data=='affectnet':
                ckpt_mae = '/data/Workspace/ABAW/code_ABAW5/checkpoints/model-20.pth'
                num_class = 9 # 9 for affectnet; 135 for emo135
            elif train_data == 'emo135':
                num_class = 135 # 9 for affectnet; 135 for emo135
                if model_name == 'vit_base_patch16':
                    ckpt_mae = '/data/Workspace/ABAW/code_ABAW5/checkpoints/mae_base_emo135.pth'
                else:
                    ckpt_mae = '/data/Workspace/ABAW/code_ABAW5/checkpoints/mae_large_emo135.pth'
            elif train_data == 'expr':
                ckpt_mae = '/data/Workspace/ABAW/code_ABAW5/checkpoints/mae_base_expr.pth'
                num_class = 8
            elif train_data == 'au':
                ckpt_mae = '/data/Workspace/ABAW/code_ABAW5/checkpoints/mae_base_au.pth'
                num_class = 12
            elif train_data == 'none':
                ckpt_mae = '/data/Workspace/ABAW/code_ABAW5/checkpoints/mae_base_none.pth'
                num_class = 12
            self.backbone_mae = getattr(models_vit, model_name)(
                            global_pool=True,
                            num_classes=num_class,
                            drop_path_rate=0.1,
                            img_size=224,
                        )
            print(f"Load pre-trained {model_name}_{train_data} checkpoint from: {ckpt_mae}")
            checkpoint = torch.load(ckpt_mae)
            checkpoint_model = checkpoint['model']
            self.backbone_mae.load_state_dict(checkpoint_model, strict=False)
            self.backbone_mae.to(device)
            for p in self.backbone_mae.parameters():
                p.requires_grad = False
        
        # #au
        # ckpt_mae = '/data/Workspace/ABAW/code_ABAW5/checkpoints/mae_base_au.pth'
        # num_class = 12
        # self.backbone_mae2 = getattr(models_vit, model_name)(
        #         global_pool=True,
        #         num_classes=num_class,
        #         drop_path_rate=0.1,
        #         img_size=224,
        #     )
        # print(f"Load pre-trained {model_name}_{train_data} checkpoint from: {ckpt_mae}")
        # checkpoint = torch.load(ckpt_mae)
        # checkpoint_model = checkpoint['model']
        # self.backbone_mae2.load_state_dict(checkpoint_model, strict=False)
        # self.backbone_mae2.to(device)
        # for p in self.backbone_mae2.parameters():
        #     p.requires_grad = False
        # #expr
        # ckpt_mae = '/data/Workspace/ABAW/code_ABAW5/checkpoints/mae_base_expr.pth'
        # num_class = 8
        # self.backbone_mae3 = getattr(models_vit, model_name)(
        # global_pool=True,
        # num_classes=num_class,
        # drop_path_rate=0.1,
        # img_size=224,
        #     )
        # print(f"Load pre-trained {model_name}_{train_data} checkpoint from: {ckpt_mae}")
        # checkpoint = torch.load(ckpt_mae)
        # checkpoint_model = checkpoint['model']
        # self.backbone_mae3.load_state_dict(checkpoint_model, strict=False)
        # self.backbone_mae3.to(device)
        # for p in self.backbone_mae3.parameters():
        #     p.requires_grad = False
                
        # self.timesnet = Timesnet(in_dim=visual_dim, out_dim=out_dim, seq_len=32).cuda()
        self.SElayer = SELayer(32)
        self.maxpool =  nn.MaxPool2d((3,1), padding=(1,0), stride=(1,1))
        self.visual_encoder = PrivateEncoder(in_size=visual_dim, hidden_size=hidden_dim, out_size=hidden_dim*2, enforce_sorted=True) # 32,64
        # self.visual_encoder = TransformerModel(64, visual_dim, nhead=8, d_hid=64, nlayers=4, dropout=0.5)
        
        # iresnet
        # vidual_dim2 = 512
        # self.resize = torchvision.transforms.Resize((112,112))
        # self.mean_mae = torch.FloatTensor([0.49895147219604985, 0.4104390648367995, 0.3656147590417074]).cuda()[None,:,None,None]
        # self.std_mae = torch.FloatTensor([0.2970847084907291, 0.2699003075660314, 0.2652599579468044]).cuda()[None,:,None,None]
        # self.mean = torch.FloatTensor([0.5] * 3).cuda()[None,:,None,None]
        # self.std = torch.FloatTensor([0.5 * 256 / 255] * 3 ).cuda()[None,:,None,None]
        # self.iresnet = insightface.iresnet100(pretrained=True)
        # self.iresnet.eval()
        # self.visual_encoder2 = PrivateEncoder(in_size=vidual_dim2, hidden_size=hidden_dim, out_size=hidden_dim*2, enforce_sorted=True)
        # for p in self.iresnet.parameters():
        #     p.requires_grad = False
            
        # expr emb
        visual_dim3 = 512
        self.backbone_emb = Pipeline_Incep(dropout_prob=0.5) #0.6
        self.backbone_emb.eval()
        self.visual_encoder3 = PrivateEncoder(in_size=visual_dim3, hidden_size=hidden_dim, out_size=hidden_dim*2, enforce_sorted=True)
        for p in self.visual_encoder3.parameters():
            p.requires_grad = False
            

        print(f'Visual backbone: {backbone_visual}')  
        if self.use_mfcc:
            audio_dim = 768
            # self.audio_encoder_mfcc = TemporalConvNet(audio_dim,[512,256,128,128],kernel_size=3, dropout=0.1)
            self.audio_encoder_mfcc = TemporalConvNet(audio_dim,[256,128],kernel_size=3, dropout=0.2)
            self.audio_transformer = TransformerModel(64, 128, nhead=8, d_hid=64, nlayers=4, dropout=0.5)
            
        audio_dim = 768
        self.audio_encoder = PrivateEncoder(in_size=audio_dim, hidden_size=hidden_dim, out_size=hidden_dim*2, enforce_sorted=True)
        # self.linear_block = LinearBlock(64, out_dim)
        audio_dim2 = 768
        self.audio_encoder2 = PrivateEncoder(in_size=audio_dim2, hidden_size=hidden_dim, out_size=hidden_dim*2, enforce_sorted=True)
        audio_dim3 = 128
        self.audio_encoder3 = PrivateEncoder(in_size=audio_dim3, hidden_size=hidden_dim, out_size=hidden_dim*2, enforce_sorted=True)
        
        text_dim = 1024
        self.text_encoder = PrivateEncoder(in_size=text_dim, hidden_size=hidden_dim, out_size=hidden_dim*2, enforce_sorted=False)

        fea_dim = 0
        if self.use_visual:
            fea_dim += hidden_dim*2
        # fea_dim = 768
        if use_audio:
            if self.use_mfcc:
                fea_dim += 64
                # fea_dim += 768
            else:
                fea_dim += hidden_dim * 2
        if use_text:
            # fea_dim += hidden_dim * 2
            fea_dim += 1024
        
        
        if self.use_multihead:
            fea_dim2 = fea_dim//8
            self.linear_block = []
            for _ in range(7):
                self.linear_block.append(predict(in_dim=fea_dim, hidden_dim=hidden_dim//2,out_dim=fea_dim2, dropout_prob=0.5).cuda()) #*2)
            self.relation_model = TransformerModel(fea_dim2*7, fea_dim2, nhead=8, d_hid=64, nlayers=4, dropout=0.5)
            self.linear2 = predict(in_dim=fea_dim2*7, hidden_dim=fea_dim2*7//8,out_dim=out_dim, dropout_prob=0.5)
        else:      
            # self.linear_block = predict(in_dim=fea_dim, hidden_dim=hidden_dim//2,out_dim=out_dim, dropout_prob=0.5) #*2
            self.linear_block = predict(in_dim=fea_dim, hidden_dim=int(hidden_dim*last_hidden_ratio),out_dim=out_dim, dropout_prob=0.5) #*2

        if self.use_cnn:
            bulid_repvgg = get_RepVGGplus_func_by_name('RepVGGplus-mine')
            self.repvgg = bulid_repvgg(deploy=False, num_classes=7, activation='tanh', in_channels=1).cuda()
        
        self.intri_relation = torch.FloatTensor([
            [ 1.        ,  0.24688253, -0.3746552 , -0.31018503, -0.27620675, -0.45377179, -0.28143648],
            [ 0.24688253,  1.        , -0.5133636 , -0.3717932 , -0.43061012, -0.60259357, -0.08106372],
            [-0.3746552 , -0.5133636 ,  1.        ,  0.14366081,  0.27002578, 0.72930203,  0.32660275],
            [-0.31018503, -0.3717932 ,  0.14366081,  1.        ,  0.31536862, 0.1588834 ,  0.15229609],
            [-0.27620675, -0.43061012,  0.27002578,  0.31536862,  1.        , 0.20849713,  0.09760578],
            [-0.45377179, -0.60259357,  0.72930203,  0.1588834 ,  0.20849713, 1.        ,  0.26904424],
            [-0.28143648, -0.08106372,  0.32660275,  0.15229609,  0.09760578, 0.26904424,  1.        ]])
    
    def forward(self, inputs_v, inputs_a, length, pretrained_feature=None, labels=None, inputs_a2=None, inputs_a3=None,
                attention_map=None, return_feature=False, use_shift=False, use_mixup=False, inputs_t=None):
        if self.use_visual:
            b, n, c, h, w = inputs_v.shape
            images_batch = torch.reshape(inputs_v, (b*n, c, h, w))
            if self.backbone_visual == 'fca':
                x = self.backbone_fca(images_batch).view(b*n, -1)
            else:
                with torch.no_grad():
                    if self.backbone_visual =='emb':
                        x = self.backbone_emb.forward_fea(images_batch)
                    else:
                        _, x = self.backbone_mae(images_batch, ret_feature=True)
                    # x += self.backbone_mae2(images_batch, ret_feature=True)[1]
                    # x += self.backbone_mae3(images_batch, ret_feature=True)[1]  
                    if self.use_patch:
                        images_batch_patch = images_batch.clone()
                        images_batch_patch[:,:,:h//2] = 0
                        x += self.backbone_mae(images_batch_patch, ret_feature=True)[1]
                        
                        images_batch_patch = images_batch.clone()
                        images_batch_patch[:,:,h//2:] = 0
                        x += self.backbone_mae(images_batch_patch, ret_feature=True)[1]

            x =  torch.reshape(x, (b, n, -1))

            if self.use_attention:
                x = torch.mul(x, attention_map[:,:,None])
            x = self.SElayer(x)
            if use_shift:
                images_batch = shift(images_batch)
            if use_mixup and labels is not None:
                x, labels, _ = mixup_data(x, labels)    
            x = self.maxpool(x)
            
            if self.use_cnn:
                return self.repvgg(x.unsqueeze(1))['main'], None, labels
        
            # vallina fusion
            x = self.visual_encoder(x, length['v'])
            # x = torch.mean(x, dim=1)
            feature = x
        
        if self.use_audio:
            if self.use_mfcc:
                x_audio = self.audio_encoder_mfcc(inputs_a.transpose(2,1)).transpose(1,2)
                x_audio = self.audio_transformer(x_audio, length['a'])
                # x_audio = torch.mean(inputs_a, dim=1)
                # x = x_audio
                x = torch.concat([x, x_audio], dim=1)
                feature=x
            else:
                x_audio = self.audio_encoder(inputs_a, length['a'])
                # x = torch.mean(inputs_a, dim=1)
                x = torch.concat([x, x_audio], dim=1)
            
            # x_audio2 = self.audio_encoder(inputs_a2, length['a2'])
            # x = torch.concatenate([x, x_audio2], dim=1)
            
            # x_audio3 = self.audio_encoder(inputs_a3, length['a3'])
            # x = torch.concatenate([x, x_audio3], dim=1)
        
        if self.use_text:
            # x_text = self.text_encoder(inputs_t, length['t'])
            x_text = torch.mean(inputs_t, dim=1)

            x = torch.concat([x, x_text], dim=1)

        if self.use_multihead:
            x_ = []
            for i in range(7):
                x_.append(self.linear_block[i](x))
            x = torch.concat(x_, dim=1)
            x = self.relation_model(torch.stack(x_,dim=1), 0)
            # x = torch.matmul(self.intri_relation[None,:].cuda(),torch.stack(x_, dim=1))
            x = self.linear2(x.view(b, -1))
        else:
            x = self.linear_block(x)
        
        out = F.sigmoid(x)
        if self.use_dual:
            out[:,:7] = 0.5*out[:,:7] + 0.5*pretrained_feature[:,:7]

        # timesnet
        # out = self.timesnet(x)
        # feature = out
        return out, feature, labels


class Vallina_fusion_visual_all(nn.Module):
    def __init__(self,in_dim=512, out_dim=139, backbone_visual='emb', use_audio=False, use_attention=False, 
                 use_dual=False):
        super(Vallina_fusion_visual_all,self).__init__()
        # mode_name = 'hubert'
        # model_4 = getattr(hub, mode_name)()
        # self.model_hubert = model_4.to(device).eval()
        device = 'cuda'  # or cpu
        self.backbone_visual = backbone_visual
        self.use_attention = use_attention
        self.use_dual = use_dual
        self.use_audio = use_audio
        hidden_dim = 128 # 32

        model_name = 'vit_base_patch16'
        train_data = 'affectnet'
        visual_dim = 768
        ckpt_mae = '/data/Workspace/ABAW/code_ABAW5/checkpoints/model-20.pth'
        num_class = 9 # 9 for affectnet; 135 for emo135
        self.backbone_mae = getattr(models_vit, model_name)(
                global_pool=True,
                num_classes=num_class,
                drop_path_rate=0.1,
                img_size=224,
            )
        print(f"Load pre-trained {model_name}_{train_data} checkpoint from: {ckpt_mae}")
        checkpoint = torch.load(ckpt_mae)
        checkpoint_model = checkpoint['model']
        self.backbone_mae.load_state_dict(checkpoint_model, strict=False)
        self.visual_encoder1 = TransformerModel(hidden_dim, visual_dim, nhead=4, d_hid=hidden_dim//2, nlayers=4, dropout=0.5)

        train_data = 'expr'
        ckpt_mae = '/data/Workspace/ABAW/code_ABAW5/checkpoints/mae_base_expr.pth'
        num_class = 8 # 9 for affectnet; 135 for emo135
        self.backbone_mae2 = getattr(models_vit, model_name)(
                global_pool=True,
                num_classes=num_class,
                drop_path_rate=0.1,
                img_size=224,
            )
        print(f"Load pre-trained {model_name}_{train_data} checkpoint from: {ckpt_mae}")
        checkpoint = torch.load(ckpt_mae)
        checkpoint_model = checkpoint['model']
        self.backbone_mae2.load_state_dict(checkpoint_model, strict=False)
        self.visual_encoder2 = TransformerModel(hidden_dim, visual_dim, nhead=4, d_hid=hidden_dim//2, nlayers=4, dropout=0.5)

        train_data = 'au'
        ckpt_mae = '/data/Workspace/ABAW/code_ABAW5/checkpoints/mae_base_au.pth'
        num_class = 12 # 9 for affectnet; 135 for emo135
        self.backbone_mae3 = getattr(models_vit, model_name)(
                global_pool=True,
                num_classes=num_class,
                drop_path_rate=0.1,
                img_size=224,
            )
        print(f"Load pre-trained {model_name}_{train_data} checkpoint from: {ckpt_mae}")
        checkpoint = torch.load(ckpt_mae)
        checkpoint_model = checkpoint['model']
        self.backbone_mae3.load_state_dict(checkpoint_model, strict=False)
        self.visual_encoder3 = TransformerModel(hidden_dim, visual_dim, nhead=4, d_hid=hidden_dim//2, nlayers=4, dropout=0.5)

        
        # self.timesnet = Timesnet(in_dim=visual_dim, out_dim=out_dim, seq_len=32).cuda()
        # self.SElayer = SELayer(32)
        # self.maxpool =  nn.MaxPool2d((3,1), padding=(1,0), stride=(1,1))
        # self.visual_encoder = PrivateEncoder(in_size=visual_dim, hidden_size=hidden_dim, out_size=hidden_dim*2, enforce_sorted=True) # 32,64
        # self.visual_encoder = TransformerModel(64, visual_dim, nhead=8, d_hid=64, nlayers=4, dropout=0.5)
        
        # iresnet
        # vidual_dim2 = 512
        # self.resize = torchvision.transforms.Resize((112,112))
        # self.mean_mae = torch.FloatTensor([0.49895147219604985, 0.4104390648367995, 0.3656147590417074]).cuda()[None,:,None,None]
        # self.std_mae = torch.FloatTensor([0.2970847084907291, 0.2699003075660314, 0.2652599579468044]).cuda()[None,:,None,None]
        # self.mean = torch.FloatTensor([0.5] * 3).cuda()[None,:,None,None]
        # self.std = torch.FloatTensor([0.5 * 256 / 255] * 3 ).cuda()[None,:,None,None]
        # self.iresnet = insightface.iresnet100(pretrained=True)
        # self.iresnet.eval()
        # self.visual_encoder2 = PrivateEncoder(in_size=vidual_dim2, hidden_size=hidden_dim, out_size=hidden_dim*2, enforce_sorted=True)
        # for p in self.iresnet.parameters():
        #     p.requires_grad = False
            
        # expr emb
        # visual_dim3 = 512
        # self.backbone_emb = Pipeline_Incep(dropout_prob=0.5) #0.6
        # self.backbone_emb.eval()
        # self.visual_encoder3 = PrivateEncoder(in_size=visual_dim3, hidden_size=hidden_dim, out_size=hidden_dim*2, enforce_sorted=True)
        # for p in self.visual_encoder3.parameters():
        #     p.requires_grad = False
            

        print(f'Visual backbone: {backbone_visual}')  
        audio_dim = 768
        self.audio_encoder = TransformerModel(hidden_dim, audio_dim, nhead=4, d_hid=hidden_dim//2, nlayers=4, dropout=0.5)
        # self.linear_block = LinearBlock(64, out_dim)
        audio_dim2 = 768
        self.audio_encoder2 = TransformerModel(hidden_dim, audio_dim2, nhead=4, d_hid=hidden_dim//2, nlayers=4, dropout=0.5)
        audio_dim3 = 128
        self.audio_encoder3 = TransformerModel(hidden_dim, audio_dim3, nhead=4, d_hid=hidden_dim//2, nlayers=4, dropout=0.5)
        
        
        fea_dim = hidden_dim 
        if use_audio:
            fea_dim += hidden_dim
        self.linear_block = predict(in_dim=fea_dim, hidden_dim=hidden_dim//2,out_dim=out_dim, dropout_prob=0.5) #*2

    def forward(self, inputs_v, inputs_a, length, pretrained_feature, labels=None, inputs_a2=None, inputs_a3=None, inputs_t=None,
                attention_map=None, return_feature=False, use_shift=False, use_mixup=False):
        b, n, c, h, w = inputs_v.shape
        images_batch = torch.reshape(inputs_v, (b*n, c, h, w))
        with torch.no_grad():
            _, x1 = self.backbone_mae(images_batch, ret_feature=True)
            _, x2 = self.backbone_mae2(images_batch, ret_feature=True)
            _, x3 = self.backbone_mae3(images_batch, ret_feature=True)
        
        x = x1 + x2 + x3
        # x1 =  torch.reshape(x1, (b, n, -1))
        # x2 =  torch.reshape(x2, (b, n, -1))
        x =  torch.reshape(x, (b, n, -1))
        x = self.visual_encoder1(x, length['v'])
        # x2 = self.visual_encoder2(x2, length['v'])
        # x3 = self.visual_encoder3(x3, length['v'])

        # x = torch.concat((x1, x2, x3), dim=1)
        
        # if self.use_attention:
        #     x = torch.mul(x, attention_map[:,:,None])
        # x = self.SElayer(x)
        # if use_shift:
        #     images_batch = shift(images_batch)
        # if use_mixup and labels is not None:
        #     x, labels, _ = mixup_data(x, labels)
        # x = self.maxpool(x)
        
        # with torch.no_grad():
        #     x = self.model_hubert(inputs)['hidden_states'][0]
        # x = self.audio_encoder(inputs, length['a'])
        
        # x = torch.mean(inputs, dim=1)

        # vallina fusion
        # x = self.visual_encoder(x, length['v'])
        feature = x

        # with torch.no_grad():
        #     inputs_iresnet = self.resize(((images_batch*self.std_mae)+self.mean_mae-self.mean)/self.std)
        #     feature_iresnet = self.iresnet(inputs_iresnet)
        # feature_iresnet = self.visual_encoder2(torch.reshape(feature_iresnet, (b, n, -1)), length['v']) 
        # x = torch.concatenate([x,feature_iresnet], dim=1)
        
        # with torch.no_grad():
        #     inputs_emb = (images_batch*self.std_mae)+self.mean_mae
        #     fea_emb = self.backbone_emb.forward_fea(inputs_emb)
        # fea_emb = self.visual_encoder3(torch.reshape(fea_emb, (b, n, -1)), length['v'])
        # x = torch.concatenate([x,fea_emb], dim=1)
        
        if self.use_audio:
            x_audio = self.audio_encoder(inputs_a, length['a'])
            x = torch.concatenate([x, x_audio], dim=1)
            
            # x_audio2 = self.audio_encoder2(inputs_a2, length['a2'])
            # x = torch.concatenate([x, x_audio2], dim=1)
            
            # x_audio3 = self.audio_encoder3(inputs_a3, length['a3'])
            # x = torch.concatenate([x, x_audio3], dim=1)

        # print(x.shape)
        x = self.linear_block(x)
        out = F.leaky_relu(x)
        if self.use_dual:
            out[:,:7] = 0.5*out[:,:7] + 0.5*pretrained_feature[:,:7]

        # timesnet
        # out = self.timesnet(x)
        # feature = out
        return out, feature, labels


def extract_mae_feature():
    ckpt_mae = '/data/Workspace/ABAW/code_ABAW5/checkpoints/model-20.pth'
    model_name = 'vit_base_patch16'
    train_data = 'affectnet'
    num_class = 9 # 9 for affectnet; 135 for emo135
    backbone_mae = getattr(models_vit, model_name)(
                    global_pool=True,
                    num_classes=num_class,
                    drop_path_rate=0.1,
                    img_size=224,
                )
    print(f"Load pre-trained {model_name}_{train_data} checkpoint from: {ckpt_mae}")
    checkpoint = torch.load(ckpt_mae)
    checkpoint_model = checkpoint['model']
    backbone_mae.load_state_dict(checkpoint_model, strict=False)
    backbone_mae = backbone_mae.cuda()
    backbone_mae.eval()

    mean = [0.49895147219604985, 0.4104390648367995, 0.3656147590417074]
    std = [0.2970847084907291, 0.2699003075660314, 0.2652599579468044]

    transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])
    splits = [
        'train',
        'val',
        'test'
        ]
    
    for split in splits:
        print("processing split:", split)
        root = f'/data/data/ABAW5/challenge4/crop_face/{split}_npy'
        save_root = f'/data/data/ABAW5/challenge4/{split}_mae'
        os.makedirs(save_root, exist_ok=True)
        vid_data = os.listdir(root)[::-1]
        for file in tqdm(vid_data, total = len(vid_data)):
            vid = file.split('.')[0]
            save_path = os.path.join(save_root, f'{vid}.npy')
            if os.path.exists(save_path):
                continue
            data = np.load(os.path.join(root, file))
            imgs = [Image.fromarray(d) for d in data]
            imgs = [transform(im) for im in imgs]
            imgs= torch.stack(imgs)
            with torch.no_grad():
                _, features = backbone_mae(imgs.cuda(), ret_feature=True)
            np.save(save_path, features.cpu().numpy())

if __name__ == '__main__':
    extract_mae_feature()
    
    # model = torch.hub.load('cfzd/FcaNet', 'fca152' ,pretrained=True)
    # inputs = torch.randn((8,3,224,224))
    # out = model(inputs)
    # print(out.shape)
    