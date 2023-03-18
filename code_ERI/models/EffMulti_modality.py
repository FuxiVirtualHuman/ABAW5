import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
# import transformers as tfs
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(curr_path, ".."))
from models.pipeline_student_InceptionResnet import Pipeline_Incep
from models.mca_net import CO_ATT
from models.cross_modal_transformers import TransformerEncoder

class textSubNet(nn.Module):
    def __init__(self):
        super(textSubNet, self).__init__()
        self.fc1 = nn.Linear(768, 64)
        # self.fc2 = nn.Linear(256, 64)

    def forward(self, bert_cls_hidden_state):
        y = F.leaky_relu(self.fc1(bert_cls_hidden_state))
        # y = F.leaky_relu(self.fc2(y))
        return y

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
        # padded_h1, _ = pad_packed_sequence(packed_h1)
        normed_h1 = layer_norm(paded_h1)
        # packed_normed_h1 = pack_padded_sequence(normed_h1, lengths.cpu())
        _, final_h2 = rnn2(normed_h1)
        return final_h1, final_h2

    def forward(self, x, lengths):
        batch_size = lengths.size(0)
        final_h1, final_h2 = self.extract_features(x, lengths, self.lstm1, self.lstm2, self.layernorm)
        # final_h1, final_h2 = self.extract_features_ptflops(x, lengths, self.lstm1, self.lstm2, self.layernorm)
        h = torch.cat((final_h1, final_h2), dim=2).permute(1,0,2).contiguous().view(batch_size, -1)
        #print(h.shape)
        #h = final_h2.view(batch_size, -1)
        y = F.leaky_relu(self.linear1(h))
        return y

class Whiten(nn.Module):
    def __init__(self):
        super(Whiten, self).__init__()

    def forward(self, x1, x2, x3):
        mean = torch.mean(torch.stack([x1, x2, x3], dim=1), dim=1)
        std = torch.std(torch.stack([x1, x2, x3], dim=1), dim=1)
        y1 = (x1 - mean)
        y2 = (x2 - mean)
        y3 = (x3 - mean)
        return [y1, y2, y3], mean

class Whiten2(nn.Module):
    def __init__(self):
        super(Whiten2, self).__init__()

    def forward(self, x1, x2, x3):
        # avt
        # mean_at = torch.mean(torch.stack([x1, x3], dim=1), dim=1)
        # mean_vt = torch.mean(torch.stack([x2, x3], dim=1), dim=1)
        # mean = torch.mean(torch.stack([mean_at, mean_vt], dim=1), dim=1)
        # y1 = (x1 - mean_at)
        # y2 = (x2 - mean_vt)
        # y3 = (x3 - mean) # method1
        # y3 = x3 # method2

        # method3
        mean = torch.mean(torch.stack([x1,x2,x3], dim=1), dim=1)
        y1 = x1 - mean
        y2 = x2 - mean
        y3 = x3
        return [y1, y2, y3], mean


class Fusion(nn.Module):

    def __init__(self, in_size, hidden_size, post_fusion_dim, output_dim):
        """
        in_size: ,64
        hidden_size: 16
        post_fusion_dim:   64
        output_dim: for cmu-mosi = 1, for iemocap = 2
        """
        super(Fusion, self).__init__()
        self.in_size = in_size # 64
        self.hidden_size = hidden_size # 16
        self.post_fusion_dim = post_fusion_dim # 64
        self.output_dim = output_dim

        self.linear_a = nn.Linear(self.in_size, self.hidden_size)
        self.linear_v = nn.Linear(self.in_size, self.hidden_size)
        self.linear_l = nn.Linear(self.in_size, self.hidden_size)

        self.linear_ap = nn.Linear(self.in_size, self.hidden_size)
        self.linear_vp = nn.Linear(self.in_size, self.hidden_size)
        self.linear_lp = nn.Linear(self.in_size, self.hidden_size)
        self.linear_mean = nn.Linear(self.in_size, self.hidden_size)

        self.p_fusion_layers_1_ = nn.Linear((self.hidden_size)*3, post_fusion_dim)
        self.p_fusion_layers_1 = nn.Linear((self.hidden_size)**3//4//4, post_fusion_dim)
        # self.p_fusion_layers_2 = nn.Linear(512, post_fusion_dim)

        self.p_fusion_layers_p1_ = nn.Linear((self.hidden_size)*3, post_fusion_dim)
        self.p_fusion_layers_p1 = nn.Linear((self.hidden_size)**3//4//4, post_fusion_dim)
        # self.p_fusion_layers_p1 = nn.Linear((self.hidden_size)**3//1, post_fusion_dim)

        # self.p_fusion_layers_p2 = nn.Linear(512, post_fusion_dim)

        self.normgate = normGate2()
        self.normgatelinear = nn.Linear(64*2,64)
        self.normgate2 = normGate2()
        # self.normgate_single = normGate_single()
        self.shared_feat = {'a':[], 't':[], 'v':[],
                            'a_de': [], 't_de': [], 'v_de': [],
                            }  # P
        self.private_feat = {'a':[], 't':[], 'v':[]}  # J
        self.J = []
        self.P = []
        self.F0 = {'fusion1':[],'fusion2':[],'mean':[]}
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.linear_all = nn.Linear(self.hidden_size, self.hidden_size * 2)

    def forward(self, a, v, l, p_a, p_v, p_l, mean, record=False):
        # input: a,v,t,pa,pv,pt,s
        # unimodal:a,v,t(Hm)
        # modality-individual: p_a, p_v, p_l
        # modality-shared: mean

        a_h = torch.sigmoid(self.linear_a(a))  # 32
        v_h = torch.sigmoid(self.linear_v(v))
        l_h = torch.sigmoid(self.linear_l(l))


        batch_size = a_h.data.shape[0]
        fusion_1 = torch.bmm(a_h.unsqueeze(2), v_h.unsqueeze(1))
        fusion_1 = self.maxpool(fusion_1)
        fusion_1 = fusion_1.view(-1, (self.hidden_size) * (self.hidden_size)//4, 1)
        fusion_1 = torch.bmm(fusion_1, l_h.unsqueeze(1))
        fusion_1 = self.maxpool(fusion_1).view(batch_size, -1)
        fusion_1 = F.leaky_relu(self.p_fusion_layers_1(fusion_1))  # J

        p_a_h = torch.sigmoid(self.linear_ap(p_a))  # 32
        p_v_h = torch.sigmoid(self.linear_vp(p_v))
        p_l_h = torch.sigmoid(self.linear_lp(p_l))

        fusion_2 = torch.bmm(p_a_h.unsqueeze(2), p_v_h.unsqueeze(1))
        fusion_2 = self.maxpool(fusion_2)
        fusion_2 = fusion_2.view(-1, (self.hidden_size) * (self.hidden_size) // 4, 1)
        fusion_2 = torch.bmm(fusion_2, p_l_h.unsqueeze(1))
        fusion_2 = self.maxpool(fusion_2).view(batch_size, -1)


        fusion_2 = F.leaky_relu(self.p_fusion_layers_p1(fusion_2))  # P

        out = self.normgate(fusion_2, mean)
        out = self.normgatelinear(out)  # 64
        out = self.normgate2(out, fusion_1)

        return out

class normGate2(nn.Module):
    def __init__(self):
        super(normGate2, self).__init__()
        running_mean = torch.zeros(64*2)
        self.register_buffer("running_mean", running_mean)
        # self.running_mean = torch.nn.Parameter(torch.zeros(dim), requires_grad = False).to('cuda')
        self.decay = 0.99

    # def norm(self, x):
        # mean = torch.mean(x, dim=0, keepdim=True)
        # weight = (x - mean)
        # weight = torch.abs(weight)
        # return weight
    def norm(self, x):
        if self.training:
            with torch.no_grad():
                mean = torch.mean(x, dim=0)
                new_mean = (1.0 - self.decay) * mean + self.decay * self.running_mean
                self.running_mean = new_mean.clone()
                # print("111111111")

        weight = (x - self.running_mean)
        weight = torch.abs(weight)
        return weight

    def forward(self, p1, m=None):
        x = torch.cat([p1, m], dim=1)
        weight = self.norm(x)
        weight = torch.sigmoid(weight)
        y = x * weight
        return y
    
class Fusion2(nn.Module):

    def __init__(self, in_size, hidden_size, post_fusion_dim, output_dim):
        """
        in_size: ,64
        hidden_size: 16
        post_fusion_dim:   64
        output_dim: for cmu-mosi = 1, for iemocap = 2
        """
        super(Fusion2, self).__init__()
        self.in_size = in_size # 64
        self.hidden_size = hidden_size # 16
        self.post_fusion_dim = post_fusion_dim # 64
        self.output_dim = output_dim
        self.pool_size=4

        self.linear_a = nn.Linear(self.in_size, self.hidden_size)
        self.linear_v = nn.Linear(self.in_size, self.hidden_size)
        self.linear_l = nn.Linear(self.in_size, self.hidden_size)

        self.linear_ap = nn.Linear(self.in_size, self.hidden_size)
        self.linear_vp = nn.Linear(self.in_size, self.hidden_size)
        self.linear_lp = nn.Linear(self.in_size, self.hidden_size)
        self.linear_mean = nn.Linear(self.in_size, self.hidden_size)

        self.p_fusion_layers_1_ = nn.Linear((self.hidden_size)*3, post_fusion_dim)
        self.p_fusion_layers_1 = nn.Linear((self.hidden_size)**3//4//4, post_fusion_dim)
        self.p_fusion_layers_1_m2 = nn.Linear(self.hidden_size**4//(self.pool_size**6), post_fusion_dim)
        self.p_fusion_layers_2_m2 = nn.Linear(self.hidden_size**4//(self.pool_size**6), post_fusion_dim)

        self.p_fusion_layers_p1_ = nn.Linear((self.hidden_size)*3, post_fusion_dim)
        self.p_fusion_layers_p1 = nn.Linear((self.hidden_size)**3//4//4, post_fusion_dim)

        self.transformer_encoder_at = TransformerEncoder(16,4,2)
        self.transformer_encoder_vt = TransformerEncoder(16,4,2)
        self.transformer_encoder_1 = TransformerEncoder(16,4,2)
        self.fc1 = nn.Linear(hidden_size, post_fusion_dim)
        self.transformer_encoder_at_p = TransformerEncoder(16,4,2)
        self.transformer_encoder_vt_p = TransformerEncoder(16,4,2)
        self.transformer_encoder_2 = TransformerEncoder(16,4,2)
        self.fc2 = nn.Linear(hidden_size, post_fusion_dim)

        self.transformer_encoder_last = TransformerEncoder(64,4,2)

        self.normgate = normGate2()
        self.normgatelinear = nn.Linear(64*2,64)
        self.normgate2 = normGate2()
        # self.normgate_single = normGate_single()

        self.maxpool = nn.MaxPool2d(self.pool_size, stride=self.pool_size)
        self.linear_all = nn.Linear(self.hidden_size, self.hidden_size * 2)

    def forward(self, a, v, l, p_a, p_v, p_l, mean, record=False):
        # input: a,v,t,pa,pv,pt,s
        # unimodal:a,v,t(Hm)
        # modality-individual: p_a, p_v, p_l
        # modality-shared: mean

        a_h = torch.sigmoid(self.linear_a(a))  # 32
        v_h = torch.sigmoid(self.linear_v(v))
        l_h = torch.sigmoid(self.linear_l(l))


        batch_size = a_h.data.shape[0]
        # fusion_at = self.transformer_encoder_at(l_h.unsqueeze(1), a_h.unsqueeze(1),a_h.unsqueeze(1))
        # fusion_vt = self.transformer_encoder_at(l_h.unsqueeze(1), v_h.unsqueeze(1),v_h.unsqueeze(1))
        # fusion_1 = self.transformer_encoder_at(fusion_at, fusion_vt, fusion_vt) + self.transformer_encoder_at(fusion_vt, fusion_at, fusion_at)
        # fusion_1 = F.leaky_relu(self.fc1(fusion_1.squeeze()))
        fusion_va = torch.bmm(a_h.unsqueeze(2), v_h.unsqueeze(1))
        fusion_va = self.maxpool(fusion_va)
        fusion_va = fusion_va.view(-1, (self.hidden_size) * (self.hidden_size) // (self.pool_size**2), 1)
        fusion_vt = torch.bmm(v_h.unsqueeze(2), l_h.unsqueeze(1))
        fusion_vt = self.maxpool(fusion_vt).view(batch_size, -1)
        fusion_vt = fusion_vt.view(-1, (self.hidden_size) * (self.hidden_size) // (self.pool_size**2), 1)

        fusion_vt = l_h*fusion_vt.squeeze()
        fusion_va = l_h*fusion_va.squeeze()
        fusion_vt = fusion_vt.unsqueeze(2)
        fusion_va = fusion_va.unsqueeze(2)
        fusion_1 = torch.bmm(fusion_va, fusion_vt.permute((0,2,1)))
        fusion_1 = self.maxpool(fusion_1).view(batch_size, -1)
        fusion_1 = F.leaky_relu(self.p_fusion_layers_1_m2(fusion_1))  # J
        # fusion_1 = torch.mean(torch.stack((l_h, fusion_1), dim=1), dim=1)
        # fusion_1 = l_h * fusion_1

        # fusion_1 = F.leaky_relu(self.p_fusion_layers_1_(torch.cat([a_h,v_h,l_h], dim=1)))
        # fusion_2 = F.leaky_relu(self.p_fusion_layers_1_(torch.mean(torch.stack([a_h, v_h, l_h], dim=2), dim=2)))

        p_a_h = torch.sigmoid(self.linear_ap(p_a))  # 32
        p_v_h = torch.sigmoid(self.linear_vp(p_v))
        p_l_h = torch.sigmoid(self.linear_lp(p_l))


        # fusion_at_p = self.transformer_encoder_at_p(p_l_h.unsqueeze(1), p_a_h.unsqueeze(1),p_a_h.unsqueeze(1))
        # fusion_vt_p = self.transformer_encoder_vt_p(p_l_h.unsqueeze(1), p_v_h.unsqueeze(1),p_v_h.unsqueeze(1))
        # fusion_2 = self.transformer_encoder_2(fusion_at_p, fusion_vt_p, fusion_vt_p) + self.transformer_encoder_2(fusion_vt_p, fusion_at_p, fusion_at_p)
        # fusion_2 = F.leaky_relu(self.fc2(fusion_2.squeeze()))

        fusion_va = torch.bmm(p_a_h.unsqueeze(2), p_l_h.unsqueeze(1))
        fusion_va = self.maxpool(fusion_va)
        fusion_va = fusion_va.view(-1, (self.hidden_size) * (self.hidden_size) // (self.pool_size**2), 1)
        fusion_vt = torch.bmm(p_v_h.unsqueeze(2), p_l_h.unsqueeze(1))
        fusion_vt = self.maxpool(fusion_vt).view(batch_size, -1)
        fusion_vt = fusion_vt.view(-1, (self.hidden_size) * (self.hidden_size) // (self.pool_size**2), 1)

        fusion_vt = l_h*fusion_vt.squeeze()
        fusion_va = l_h*fusion_va.squeeze()
        fusion_vt = fusion_vt.unsqueeze(2)
        fusion_va = fusion_va.unsqueeze(2)

        fusion_2 = torch.bmm(fusion_va, fusion_vt.permute((0,2,1)))
        fusion_2 = self.maxpool(fusion_2).view(batch_size, -1)
        fusion_2 = F.leaky_relu(self.p_fusion_layers_2_m2(fusion_2))  # J
        # fusion_2 = torch.mean(torch.stack((p_l_h, fusion_2), dim=1), dim=1)
        # fusion_2 = p_l_h*fusion_2
        #
        # fusion_2 = torch.bmm(p_a_h.unsqueeze(2), p_v_h.unsqueeze(1))
        # fusion_2 = self.maxpool(fusion_2)
        # fusion_2 = fusion_2.view(-1, (self.hidden_size) * (self.hidden_size) // 4, 1)
        # fusion_2 = torch.bmm(fusion_2, p_l_h.unsqueeze(1))
        # fusion_2 = self.maxpool(fusion_2).view(batch_size, -1)
        #
        # fusion_2 = F.leaky_relu(self.p_fusion_layers_p1(fusion_2))  # P

        # S: mean; P: fusion2; J: fusion1

        # out = self.normgate(fusion_2, mean)
        # out = self.normgatelinear(out)  # 64
        # out = self.normgate2(out, fusion_1)
        # out = self.transformer_encoder_last(fusion_1.unsqueeze(1), fusion_2.unsqueeze(1), mean.unsqueeze(1))
        out = torch.mean(torch.stack((fusion_1, fusion_2, mean),dim=1),dim=1)

        return out.squeeze()

class Fusion_co_att(nn.Module):

    def __init__(self, in_size, hidden_size, post_fusion_dim, output_dim):
        """
        in_size: ,64
        hidden_size: 16
        post_fusion_dim:   64
        output_dim: for cmu-mosi = 1, for iemocap = 2
        """
        super(Fusion_co_att, self).__init__()
        self.in_size = in_size  # 64
        self.hidden_size = hidden_size  # 16
        self.post_fusion_dim = post_fusion_dim  # 64
        self.output_dim = output_dim

        self.co_att1_a = CO_ATT(64,2,32)
        self.co_att1_v = CO_ATT(64,2,32)
        self.co_att1_t = CO_ATT(64,2,32)
        self.co_att2_a = CO_ATT(64,2,32)
        self.co_att2_v = CO_ATT(64,2,32)
        self.co_atts_t = CO_ATT(64,2,32)

        self.linear_a = nn.Linear(self.in_size, self.hidden_size)
        self.linear_v = nn.Linear(self.in_size, self.hidden_size)
        self.linear_l = nn.Linear(self.in_size, self.hidden_size)

        self.linear_ap = nn.Linear(self.in_size, self.hidden_size)
        self.linear_vp = nn.Linear(self.in_size, self.hidden_size)
        self.linear_lp = nn.Linear(self.in_size, self.hidden_size)
        self.linear_mean = nn.Linear(self.in_size, self.hidden_size)

        self.p_fusion_layers_1_ = nn.Linear((self.hidden_size) , post_fusion_dim)
        self.p_fusion_layers_1 = nn.Linear((self.hidden_size) ** 3 // 4 // 4, post_fusion_dim)
        # self.p_fusion_layers_2 = nn.Linear(512, post_fusion_dim)

        self.p_fusion_layers_p1_ = nn.Linear((self.hidden_size), post_fusion_dim)
        self.p_fusion_layers_p1 = nn.Linear((self.hidden_size) ** 3 // 4 // 4, post_fusion_dim)
        # self.p_fusion_layers_p1 = nn.Linear((self.hidden_size)**3//1, post_fusion_dim)

        # self.p_fusion_layers_p2 = nn.Linear(512, post_fusion_dim)

        self.normgate = normGate2()
        self.normgatelinear = nn.Linear(64 * 2, 64)
        self.normgate2 = normGate2()
        # self.normgate_single = normGate_single()
        self.shared_feat = {'a': [], 't': [], 'v': [],
                            'a_de': [], 't_de': [], 'v_de': [],
                            }  # P
        self.private_feat = {'a': [], 't': [], 'v': []}  # J
        self.J = []
        self.P = []
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.linear_all = nn.Linear(self.hidden_size, self.hidden_size * 2)

    def forward(self, a, v, l, p_a, p_v, p_l, mean, record=False):
        # input: a,v,t,pa,pv,pt,s
        # unimodal:a,v,t(Hm)
        # modality-individual: p_a, p_v, p_l
        # modality-shared: mean

        a_h = torch.sigmoid(self.linear_a(a))  # 32
        v_h = torch.sigmoid(self.linear_v(v))
        l_h = torch.sigmoid(self.linear_l(l))

        batch_size = a_h.data.shape[0]
        a_h = self.co_att1_a(a_h, v_h, l_h)
        v_h = self.co_att1_a(v_h, a_h, l_h)
        l_h = self.co_att1_a(l_h, v_h, a_h)

        fusion_1 = torch.mean(torch.stack([a_h, v_h,l_h], dim=2), dim=2)
        fusion_1 = F.leaky_relu(self.p_fusion_layers_1_(fusion_1))
        # fusion_1 = F.leaky_relu(self.p_fusion_layers_1_(torch.cat([a_h, v_h, l_h], dim=1)))
        # fusion_2 = F.leaky_relu(self.p_fusion_layers_1_(torch.mean(torch.stack([a_h, v_h, l_h], dim=2), dim=2)))

        p_a_h = torch.sigmoid(self.linear_ap(p_a))  # 32
        p_v_h = torch.sigmoid(self.linear_vp(p_v))
        p_l_h = torch.sigmoid(self.linear_lp(p_l))

        p_a_h = self.co_att2_a(p_a_h, p_v_h, p_l_h)
        p_v_h = self.co_att2_a(p_v_h, p_l_h, p_a_h)
        p_l_h = self.co_att2_a(p_l_h, p_v_h, p_a_h)
        fusion_2 = torch.mean(torch.stack([p_a_h, p_v_h, p_l_h], dim=2), dim=2)
        fusion_2 = F.leaky_relu(self.p_fusion_layers_p1_(fusion_2))
        # fusion_2 = F.leaky_relu(self.p_fusion_layers_p1_(torch.mean(torch.stack([p_a_h,p_v_h, p_l_h],dim=2), dim=2)))

        out = self.normgate(fusion_2, mean)
        out = self.normgatelinear(out)  # 64
        out = self.normgate2(out, fusion_1)

        return out


class predict(nn.Module):

    def __init__(self, post_fusion_dim, output_dim, dropout_prob=0.1):

        super().__init__()
        self.post_fusion_dim = post_fusion_dim
        self.output_dim = output_dim
        self.pre_layer_2 = nn.Linear(self.post_fusion_dim, 16)
        self.pre_layer_4 = nn.Linear(16, self.output_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, out):

        out = self.dropout(out)
        out =  F.leaky_relu(self.pre_layer_2(out))
        # out = F.normalize(out, dim=1)

        out = self.pre_layer_4(out)
        return out


class EffMulti_modality2(nn.Module):

    def __init__(self, out_dim=6):
        super(EffMulti_modality2, self).__init__()
        self.model_emb = Pipeline_Incep(dropout_prob=0.6) #0.6


        self.audio_encoder = PrivateEncoder(in_size=1024, hidden_size=32, out_size=64, enforce_sorted=False)
        self.video_encoder = PrivateEncoder(in_size=512, hidden_size=32, out_size=64, enforce_sorted=False)
        self.text_encoder = PrivateEncoder(in_size=1024, hidden_size=32, out_size=64, enforce_sorted=False)
        self.fusion = Fusion2(in_size=64, hidden_size=16, post_fusion_dim=64, output_dim=1)

        self.whiten = Whiten2()
        self.predict_fin = predict(post_fusion_dim=64, output_dim=out_dim, dropout_prob=0.5) #*2

    def batch_based_mean(self, x):
        """
        sub the batch-based mean of x
        """
        m = torch.mean(x, dim=0)
        x = x - m
        return x

    def forward(self, images, text_x ,audio_x, lengths, record=False):
        # emb_model
        b, n, c, h, w = images.shape
        images_batch = torch.reshape(images, (b*n, c, h, w))
        video_x = self.model_emb.forward_fea(images_batch)
        video_x = torch.reshape(video_x, (b, n, -1))
        # private_encoder
        
        video_p = self.video_encoder(video_x, lengths['v'].cuda())
        audio_p = self.audio_encoder(audio_x, lengths['a'].cuda())
        text_p = self.text_encoder(text_x, lengths['t'].cuda())

        [audio_p1, video_p1, text_p1], means = self.whiten(audio_p, video_p, text_p)

        p_fusion = self.fusion(audio_p, video_p, text_p, audio_p1, video_p1, text_p1, means, record)
        out = self.predict_fin(p_fusion)

        return out
    
if __name__ == '__main__':
    model = EffMulti_modality2().cuda()
    images = torch.randn((2,4,3,224,224)).cuda()
    text_x = torch.randn((2,4,1024)).cuda()
    audio_x = torch.randn((2,4,1024)).cuda()
    lengths = {'v':torch.LongTensor([2,2]),'a':torch.LongTensor([2,2]),'t':torch.LongTensor([2,2]),}
    pretrained = '20230213-135952_best'
    if pretrained:
        checkpoint = torch.load(os.path.join(f'/data/Workspace/ABAW/code_ABAW5/checkpoints', f'{pretrained}.pt'))
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded pretrained model {}".format(pretrained))
   
    # with torch.no_grad():
    # model.eval()
    # out2 = model(images, text_x, audio_x, lengths)
    # print(out2)
     
    model.train()
    out1 = model(images, text_x, audio_x, lengths)
    print(out1)
    
