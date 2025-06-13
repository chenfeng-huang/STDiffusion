import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import SelfAttention, DiffusionEmbedding, WaveletModule, SeasonEncoder, SeasonDecoder, TrendEncoder, TrendDecoder, CrossAttention
from einops import reduce, rearrange
from RevIN import RevIN
from LearnableMovingAvg import LearnableMovingAvg

class ScoreNetwork(nn.Module):
    def __init__(self, config, n_feature, seq_len, device):
        super().__init__()
        self.device = device
        self.channels = config['channels']
        self.seq_len = seq_len
        self.n_feature = n_feature 
        self.att_mask = config['season_att_mask']
        
        self.season_block = SeasonBlock(
            nheads=config['nheads'],
            channels=self.channels,   
            freq_tier = config['freq_tier'],
            res_layers=config['res_layers'],
            season_att=config['season_att'],
            att_mask = config['season_att_mask'],
            device=device
        )

        self.trend_block = TrendBlock(
            self.channels, 
            config['res_layers'], 
            seq_len,
            config['revin_affine']
        )

        self.s_enc = SeasonEncoder(
            self.channels, 
            n_feature, 
            config['nheads'],
            config['s_enc_att'],
            config['enc_att_mask'],
        )
        self.s_dec = SeasonDecoder(self.channels, n_feature)
        self.t_enc = TrendEncoder(
            self.channels, 
            n_feature, 
            config['nheads'],
            config['t_enc_att'], 
            config['enc_att_mask'],
        )
        self.t_dec = TrendDecoder(self.channels, n_feature)
        self.l_mve_avg = LearnableMovingAvg(
            num_features=n_feature, 
            affine=config['lma_affine']
            ).to(self.device)
        self.ts_correlation = SeasonalTrendCorrelation(
            self.channels, 
            config['nheads'],
            config['corr_att'],
            config['corr_att_mask'], 
        )

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config['num_steps'],
            embedding_dim=config['diffusion_embedding_dim'],
            projection_dim=self.channels
        )
        
    


    def forward(self, x, diffusion_step):
        diffusion_emb = self.diffusion_embedding(diffusion_step) # (B, C, 1)
        '''De-trend'''
        season, trend = self.l_mve_avg(x, 'deTrend') # (B, L0, K), (B, L0, K)
    
        '''Encoder'''
        season = self.s_enc(season, diffusion_emb) # (B, C, L0)
        trend = self.t_enc(trend, diffusion_emb)

        '''Trend Analysis'''
        trend = self.trend_block(trend)
      
        '''Season Analysis'''
        season = self.season_block(season)

        '''Trend Season Correlation'''
        trend, season = self.ts_correlation(trend, season)
        #trend, season = trend.permute(0, 2, 1), season.permute(0, 2, 1)
        '''Decoder'''
        season = self.s_dec(season.permute(0, 2, 1)) # (B, L0, K)
        trend = self.t_dec(trend.permute(0, 2, 1))

        '''Restore'''
        y, _ = self.l_mve_avg(season, 'restore', trend)

        return y  # (B, C, L0)
        
class SeasonBlock(nn.Module):
    def __init__(self, nheads, channels, freq_tier, res_layers, season_att, att_mask, device):
        super().__init__()
        self.att_layers = nn.ModuleList([
            SelfAttention(
                heads=nheads,  
                layers=1,
                channels=channels,
                type=season_att,
                mask=att_mask
                ) for _ in range(res_layers)
        ])
        
        self.wavelet_module = WaveletModule(freq_tier)

        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels),
            nn.Dropout(0.1)
        )
        self.ln = nn.LayerNorm(channels, elementwise_affine=False)
        
    def forward(self, x):
        B, C, L = x.shape
        freq_collect = self.wavelet_module.decomposite(x) # (B, C, Li) * J
        freq_out_collect = []
        for tier in range(len(freq_collect)):   
            freq = freq_collect[tier].permute(0, 2, 1) # (B, Li, C)
            for att in self.att_layers:
                freq = freq + att(freq)
                freq = freq + self.mlp(self.ln(freq))

            freq_out_collect.append(freq.permute(0, 2, 1))

        y = self.wavelet_module.reconstructe(freq_out_collect) # (B, C, L0)

        return y
    

class TrendBlock(nn.Module):
    def __init__(self, channels, res_layers, seq_len, affine):
        super().__init__()
        self.seq_len = seq_len
        self.channels = channels
        self.revin = RevIN(self.channels, affine=affine)

        self.mlp_layers = nn.ModuleList([
             nn.Sequential(
                nn.Linear(self.seq_len, self.channels),
                nn.GELU(),
                nn.Linear(self.channels, self.channels*4),
                nn.GELU(),
                nn.Linear(self.channels*4, self.channels),
                nn.GELU(),
                nn.Linear(self.channels, self.seq_len),
                nn.Dropout(0.1)
            )    for _ in range(res_layers)
        ])
        #self.ln = nn.LayerNorm(channels, elementwise_affine=False)
   
    def forward(self, x):
        B, C, L0 = x.shape
        #x = x.permute(0, 2, 1)
        x = self.revin(x.permute(0, 2, 1), 'norm').permute(0, 2, 1)

        for mlp in self.mlp_layers:
            x = x + mlp(x)

        #x = x.permute(0, 2, 1)

        x = self.revin(x.permute(0, 2, 1), 'denorm').permute(0, 2, 1)

        return x

class SeasonalTrendCorrelation(nn.Module):
    def __init__(self, channels, nheads, corr_att, att_mask):
        super().__init__()
        self.trend_projection = nn.Sequential(nn.Linear(channels, 2 * channels), nn.GELU())
        self.season_projection = nn.Sequential(nn.Linear(channels, 2 * channels), nn.GELU())

        self.crs_att_t = CrossAttention(
            heads=nheads, layers=1, channels=channels, type=corr_att, mask=att_mask
        )
        self.crs_att_s = CrossAttention(
            heads=nheads, layers=1, channels=channels, type=corr_att, mask=att_mask
        )
        
        self.trend_mlp = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels),
            nn.Dropout(0.1)
        )
        self.season_mlp = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels),
            nn.Dropout(0.1)
        )
   
        self.ln_s = nn.LayerNorm(channels, elementwise_affine=False)
        self.ln_t = nn.LayerNorm(channels, elementwise_affine=False)
       
    def forward(self, trend, season):
        trend = self.trend_projection(trend.permute(0, 2, 1))
        trend, trend_cond = torch.chunk(trend, 2, dim=-1)
        
        
        season = self.season_projection(season.permute(0, 2, 1))
        season, season_cond = torch.chunk(season, 2, dim=-1)

        trend = trend + self.crs_att_t(trend, season_cond)
        trend = trend + self.trend_mlp(self.ln_t(trend))

        season = season + self.crs_att_s(season, trend_cond)
        season = season + self.season_mlp(self.ln_s(season))

        return trend, season
