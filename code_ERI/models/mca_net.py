# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

import torch.nn as nn
import torch.nn.functional as F
import torch, math


# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------

class FC(nn.Module):
    def __init__(self, in_size, out_size, dropout_r=0., use_relu=True):
        super(FC, self).__init__()
        self.dropout_r = dropout_r
        self.use_relu = use_relu

        self.linear = nn.Linear(in_size, out_size)

        if use_relu:
            self.relu = nn.ReLU(inplace=True)

        if dropout_r > 0:
            self.dropout = nn.Dropout(dropout_r)

    def forward(self, x):
        x = self.linear(x)

        if self.use_relu:
            x = self.relu(x)

        if self.dropout_r > 0:
            x = self.dropout(x)

        return x

class MLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size, dropout_r=0., use_relu=True):
        super(MLP, self).__init__()

        self.fc = FC(in_size, mid_size, dropout_r=dropout_r, use_relu=use_relu)
        self.linear = nn.Linear(mid_size, out_size)

    def forward(self, x):
        return self.linear(self.fc(x))


class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps

        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class MHAtt(nn.Module):
    def __init__(self, HIDDEN_SIZE,MULTI_HEAD,HIDDEN_SIZE_HEAD,DROPOUT_R=0.5 ):
        super(MHAtt, self).__init__()
        self.HIDDEN_SIZE_HEAD = HIDDEN_SIZE_HEAD
        self.MULTI_HEAD = MULTI_HEAD
        self.HIDDEN_SIZE = HIDDEN_SIZE

        self.linear_v = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.linear_k = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.linear_q = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.linear_merge = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)

        self.dropout = nn.Dropout(DROPOUT_R)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.MULTI_HEAD,
            self.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.MULTI_HEAD,
            self.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.MULTI_HEAD,
            self.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.HIDDEN_SIZE
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)


# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    def __init__(self, HIDDEN_SIZE,FF_SIZE):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=HIDDEN_SIZE,
            mid_size=FF_SIZE,
            out_size=HIDDEN_SIZE,
            dropout_r=0.5,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)


# ------------------------
# ---- Self Attention ----
# ------------------------

class SA(nn.Module):
    def __init__(self, HIDDEN_SIZE,MULTI_HEAD,HIDDEN_SIZE_HEAD,FF_SIZE):
        super(SA, self).__init__()

        self.mhatt = MHAtt(HIDDEN_SIZE,MULTI_HEAD,HIDDEN_SIZE_HEAD,)
        self.ffn = FFN(HIDDEN_SIZE,FF_SIZE)

        self.dropout1 = nn.Dropout(0.5)
        self.norm1 = LayerNorm(HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(0.5)
        self.norm2 = LayerNorm(HIDDEN_SIZE)

    def forward(self, x, x_mask):
        x = self.norm1(x + self.dropout1(
            self.mhatt(x, x, x, x_mask)
        ))

        x = self.norm2(x + self.dropout2(
            self.ffn(x)
        ))

        return x


# -------------------------------
# ---- Self Guided Attention ----
# -------------------------------

class SGA(nn.Module):
    def __init__(self, HIDDEN_SIZE, MULTI_HEAD,HIDDEN_SIZE_HEAD,FF_SIZE):
        super(SGA, self).__init__()

        self.mhatt1 = MHAtt(HIDDEN_SIZE,MULTI_HEAD,HIDDEN_SIZE_HEAD,)
        self.mhatt2 = MHAtt(HIDDEN_SIZE,MULTI_HEAD,HIDDEN_SIZE_HEAD,)
        self.ffn = FFN(HIDDEN_SIZE,FF_SIZE)

        self.dropout1 = nn.Dropout(0.5)
        self.norm1 = LayerNorm(HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(0.5)
        self.norm2 = LayerNorm(HIDDEN_SIZE)

        self.dropout3 = nn.Dropout(0.5)
        self.norm3 = LayerNorm(HIDDEN_SIZE)

    def forward(self, x, y, x_mask, y_mask):
        x = self.norm1(x + self.dropout1(
            self.mhatt1(x, x, x, x_mask)
        ))

        x = self.norm2(x + self.dropout2(
            self.mhatt2(y, y, x, y_mask)
        ))

        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))

        return x

# ------------------------------------------------
# ---- MAC Layers Cascaded by Encoder-Decoder ----
# ------------------------------------------------

class MCA_ED(nn.Module):
    def __init__(self, HIDDEN_SIZE,MULTI_HEAD,HIDDEN_SIZE_HEAD,FF_SIZE):
        super(MCA_ED, self).__init__()

        self.enc_list = nn.ModuleList([SA(HIDDEN_SIZE,MULTI_HEAD,HIDDEN_SIZE_HEAD,FF_SIZE) for _ in range(2)])
        self.dec_list = nn.ModuleList([SGA(HIDDEN_SIZE,MULTI_HEAD,HIDDEN_SIZE_HEAD,FF_SIZE) for _ in range(2)])

    def forward(self, x, y):
        # Get hidden vector
        x_mask = None
        y_mask = None


        for enc in self.enc_list:
            x = enc(x, x_mask)

        for dec in self.dec_list:
            y = dec(y, x, y_mask, x_mask)

        return x, y

    # Masking
    def make_mask(self, feature):
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0).unsqueeze(1).unsqueeze(2)

class CO_ATT(nn.Module):
    def __init__(self, HIDDEN_SIZE, MULTI_HEAD,HIDDEN_SIZE_HEAD):
        super(CO_ATT, self).__init__()

        self.mhatt1 = MHAtt(HIDDEN_SIZE,MULTI_HEAD,HIDDEN_SIZE_HEAD,)

        self.norm1 = LayerNorm(HIDDEN_SIZE)
        self.dropout1 = nn.Dropout(0.5)

    def forward(self, x, y, z, y_mask=None ):
        x = self.norm1(x + self.dropout1(self.mhatt1(z, y, x, y_mask)).squeeze())

        return x

if __name__ == '__main__':
    net = CO_ATT(256,2,128)
    x = torch.randn((4,256))
    y = torch.randn((4,256))
    z = torch.randn((4,256))
    out = net(x,y,z)
    print(out)