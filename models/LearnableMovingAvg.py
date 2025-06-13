import torch
import torch.nn as nn

class MovingAvg(nn.Module):
    def __init__(self, kernel_size, stride):
        self.strid = stride
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        x = x.repeat_interleave(self.strid, dim=1)
        return x

class LearnableMovingAvg(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LearnableMovingAvg, self).__init__()
        self.eps = eps
        self.num_features = num_features
        self.affine = affine
        if self.affine:
            self._init_params()

        self.kernel_size = [1, 2, 4, 6, 12]
        self.moving_avg = [MovingAvg(kernel, stride=1) for kernel in self.kernel_size]
        self.mlp = torch.nn.Linear(1, len(self.kernel_size))
        self.input_collect = []
        self.trend_input = []
        self.season_input = []
        self.trend_pred = []
        self.season_pred = []
        self.collect_limit = 0
        #self.moving_avg = MovingAvg(3, stride=1)


    def forward(self, x, mode:str, trend=None):
        if mode == 'deTrend':
            y, trend = self._normalize(x)
            
        elif mode == 'restore':
            y = self._denormalize(x, trend)
        else: raise NotImplementedError
        return y, trend

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _normalize(self, x):
        if len(self.input_collect) < self.collect_limit:
            self.input_collect.append(x)
        trend = []
        for ma_block in self.moving_avg:
            ma = ma_block(x)
            trend.append(ma.unsqueeze(-1))
        trend = torch.cat(trend, dim=-1)
        trend = torch.sum(trend * nn.Softmax(-1)(self.mlp(x.unsqueeze(-1))), dim=-1)
        if len(self.trend_input) < self.collect_limit:
            self.trend_input.append(trend)
        
        if self.affine:
            trend = trend * self.affine_weight
            trend = trend + self.affine_bias

        season = x - trend
        if len(self.season_input) < self.collect_limit:
            self.season_input.append(season)
        return season, trend

    def _denormalize(self, x, trend):
        if self.affine:
            trend = trend - self.affine_bias
            trend = trend / (self.affine_weight + self.eps*self.eps)
        
        if len(self.trend_pred) < self.collect_limit:
            self.trend_pred.append(trend)
            self.season_pred.append(x)

        y = trend + x

        return y
