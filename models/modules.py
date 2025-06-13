import torch.nn as nn
import torch
from LearnableDWT.transform1d import DWT1DForward, DWT1DInverse, DWT1D
# from adaptive_wavelets.transform1d import DWT1d
import torch.nn.functional as F
from Attention import FourierAttention, FullAttention


class SeasonEncoder(nn.Module):
    def __init__(self, channels, feature_dim, nheads, att_type, att_mask):
        super().__init__()
        self.channels = channels
        self.conv = nn.Sequential(
            nn.Conv1d(feature_dim, channels, 3, 1, 1),
            nn.GELU()
        )

        self.ln1 = nn.LayerNorm(channels, elementwise_affine=False)
        self.ln2 = nn.LayerNorm(channels, elementwise_affine=False)
        self.att = SelfAttention(
            heads=nheads, layers=1, channels=channels, type=att_type, mask=att_mask
        )
    
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels),
            nn.GELU()
        )
       
        
    def forward(self, x, diffusion_emb):
        B, L, K = x.shape
        x = self.conv(x.permute(0, 2, 1)) #(B, C, L)

        emb = diffusion_emb.unsqueeze(1)
        
        
        scale, shift = torch.chunk(emb, 2, dim=2)
        x = self.ln1(x.permute(0, 2, 1)) * (1 + scale) + shift  # (B, L0, C)
        x = x + self.att(x)
        x = x + self.mlp(self.ln2(x))
        x = x.permute(0, 2, 1)

        return x

class SeasonDecoder(nn.Module):
    def __init__(self, channels, feature_dim):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, feature_dim),
        )
    def forward(self, x):
        B, C, L = x.shape
        x = self.mlp(x.permute(0, 2, 1))

        return x

class TrendEncoder(nn.Module):
    def __init__(self, channels, feature_dim, nheads, att_type, att_mask):
        super().__init__()
        self.channels = channels
        self.conv = nn.Sequential(
            nn.Conv1d(feature_dim, channels, 3, 1, 1),
            nn.GELU()
        )

        self.ln1 = nn.LayerNorm(channels, elementwise_affine=False)
        self.ln2 = nn.LayerNorm(channels, elementwise_affine=False)
        self.att = SelfAttention(
            heads=nheads, layers=1, channels=channels, type=att_type, mask=att_mask
        )
    
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels),
            nn.GELU()
        )
 

    def forward(self, x, diffusion_emb):
        B, L, K = x.shape
        x = self.conv(x.permute(0, 2, 1))#(B, C, L)

        emb = diffusion_emb.unsqueeze(1)


        scale, shift = torch.chunk(emb, 2, dim=2)
        x = self.ln1(x.permute(0, 2, 1)) * (1 + scale) + shift  # (B, L0, C)
        x = x + self.att(x)
        x = x + self.mlp(self.ln2(x))
        x = x.permute(0, 2, 1)
        

        return x

class TrendDecoder(nn.Module):
    def __init__(self, channels, feature_dim):
        super().__init__()
        # self.conv = nn.Conv1d(channels, feature_dim, 1)
        # self.relu = nn.ReLU()  
        # self.gelu = nn.GELU()
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, feature_dim),
        )
        #self.pe = LearnablePositionalEncoding(channels)
    def forward(self, x):
        B, C, L = x.shape
        #x = self.pe(x)
        x = self.mlp(x.permute(0, 2, 1))
        # x = self.conv(x)
        # x = self.gelu(x)
        # x = x.permute(0, 2, 1)

        return x
            
    

class WaveletModule(nn.Module):
    def __init__(self, freq_tier):
        super().__init__()
        # self.dwt = DWT1DForward(wave='db3', mode='periodic', J=freq_tier)
        # self.idwt = DWT1DInverse(wave='db3', mode='periodic')
        self.wt = DWT1D(wave='db3', mode='periodic', J=freq_tier)

    def decomposite(self, x):
        #xl, xh = self.dwt(x)
        #import ipdb; ipdb.set_trace()
        xl, xh = self.wt.decompose(x)
        x_collect = [] 
        for xh_det in xh:
            x_collect.append(xh_det)

        x_collect.append(xl)

        return x_collect
    
    def reconstructe(self, x_out_collect):
        xh = []
        for i in range(len(x_out_collect)-1):
            xh.append(x_out_collect[i])

        xl = x_out_collect[-1]
        
        #return self.idwt((xl, xh)) 
        return self.wt.reconstruct((xl, xh))
    
        
class SelfAttention(nn.Module):
    def __init__(self, heads, layers, channels, type, mask):
        super().__init__()
        if type == 'full':
            # encoder_layer = nn.TransformerEncoderLayer(
            #     d_model=channels, 
            #     nhead=heads, 
            #     dim_feedforward=8, 
            #     activation='gelu', 
            #     batch_first=True
            # )
            # self.att_layer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
            att = FullAttention(mask_flag=mask)
            #self.att_layer = AttentionLayer(att, channels, heads)
            self.att_layers = nn.ModuleList([
                AttentionLayer(
                att, channels, heads
                ) for _ in range(layers)
            ])
        elif type == 'fourier':
            att = FourierAttention()
            self.att_layers = nn.ModuleList([
                AttentionLayer(
                    att, channels, heads
                    ) for _ in range(layers)
            ])
        self.ln = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(0.1)


    def forward(self, x):
        for att in self.att_layers:
            x = x + att(x, x, x)
            x = x + self.dropout(x)
            x = self.ln(x)
    
        return x
    
class CrossAttention(nn.Module):
    def __init__(self, heads, layers, channels, type, mask):
        super().__init__()
        if type == 'full':
            # encoder_layer = nn.TransformerEncoderLayer(
            #     d_model=channels, 
            #     nhead=heads, 
            #     dim_feedforward=8, 
            #     activation='gelu', 
            #     batch_first=True
            # )
            # self.att_layer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
            att = FullAttention(mask_flag=mask)
            self.att_layer = AttentionLayer(att, channels, heads)
        elif type == 'fourier':
            att = FourierAttention()
            self.att_layer = AttentionLayer(att, channels, heads)
        self.ln = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(0.1)


    def forward(self, x, x_cond):
        y = self.att_layer(x, x_cond, x_cond)
        x = x + self.dropout(y)
        x = self.ln(x)
    
        return x
    
class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads=8, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, _ = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out)

    

class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim*2)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table

