import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ScoreModel import ScoreNetwork
from tqdm import tqdm



class DiffModel(nn.Module):
    def __init__(self, dataset_info, config, device):
        super().__init__()
        self.device = device

        self.config_train = config['train']
        self.config_score = config['score_network']
        self.config_diff = config['diff_model']

        self.features = dataset_info['features']
        self.nsample = dataset_info['nsample']
        self.dataset_info = dataset_info
        self.batch_size = self.config_train['batch_size']
        self.sample_size = self.config_diff['sample_size']
        self.batch_count = self.nsample//self.sample_size + 1
        self.seq_len = self.config_train['seq_len']
        
        self.is_conditional = self.config_diff['is_conditional']
        
        
        self.score_model = ScoreNetwork(
            self.config_score, len(self.features), self.seq_len, self.device
        ) 
        

        self.criterion = nn.MSELoss(reduction='mean')

        self.num_steps = self.config_score['num_steps']
        self.sample_steps = self.config_diff['sample_steps']
        if self.config_diff['schedule'] == 'quad':
            self.beta = np.linspace(
                self.config_diff['beta_start'] ** 0.5, self.config_diff['beta_end'] ** 0.5, self.num_steps
            ) ** 2
           
        elif self.config_diff['schedule'] == 'linear':
            self.beta = np.linspace(
                self.config_diff['beta_start'], self.config_diff['beta_end'], self.num_steps
            )

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)
       
    def _normalize_to_neg_one_to_one(self, x):
        return x * 2 - 1

    def _unnormalize_to_zero_to_one(self, x):
        return (x + 1) * 0.5
    
    def _get_mask(self, x, pred_len):
        mask = torch.ones_like(x)
        mask[:, -pred_len:, :] = 0.
        
        return mask
    
    
    def forward(self, x):
        x = self._normalize_to_neg_one_to_one(x)
        x0 = x.clone()
        
        B, L0, K = x.size() 
        '''Add Noise'''
        s = torch.randint(0, self.num_steps, [B]).to(self.device)
        current_alpha = self.alpha_torch[s]  # (B, 1, 1)
        noise = torch.randn_like(x)
        x = (current_alpha ** 0.5) * x + (1.0 - current_alpha) ** 0.5 * noise # adding noise 

      
        '''Score Model Prediction'''
        x_pred = self.score_model(x, s) 

        '''Loss'''
    
        loss = self.criterion(x_pred, noise)
        

        return loss
 
    @torch.no_grad()
    def generate(self):
        sample_result = []
        '''Denoise to generate samples'''    
        i = 0
        resample_count = 0
        with tqdm(total=self.batch_count, desc="Generating samples") as pbar:
            while i < self.batch_count:
                min_val = 0
                max_val = 1
                noise = torch.randn(self.sample_size, self.seq_len, len(self.features)).to(self.device)
                sample = self._denoise(noise)
       
                if torch.any(sample < -1):
                    print('val<-1:', torch.sum(sample < -1))
        
                    min_val = min(sample[sample < -1])
                    print('min:', min_val)
                    if min_val < -2: 
                        print('Resample')
                        resample_count += 1
                        continue
                    

                if torch.any(sample > 1):
                    print('val>1:', torch.sum(sample > 1))

                    max_val = max(sample[sample > 1])
                    print('max:', max_val)
                    if max_val > 2: 
                        print('Resample')
                        resample_count += 1
                        continue
                        

                if torch.any(torch.isnan(sample)):
                    print('NaN value count:', torch.sum(torch.isnan(sample)))
                    print('Resample')
                    resample_count += 1
                    continue
      
                sample = self._unnormalize_to_zero_to_one(sample)
  
                sample_result.append(sample)
                i += 1  # Increment the loop counter only if the iteration is successful
                pbar.update(1)  
        print('resample count:', resample_count)

        

        return torch.cat(sample_result, dim=0)
  
    @torch.no_grad()
    def _denoise(self, x_s):

        for s in tqdm(range(self.sample_steps - 1, -1, -1), desc="Denoising"):

            '''Generating'''
            x_pred = self.score_model(x_s, torch.tensor([s])) 

            # if torch.isnan(x_pred).any():
            #     print(s)
            #     import ipdb; ipdb.set_trace()
            #print(x_pred)
            '''Denoise'''
            coeff1 = 1 / self.alpha_hat[s] ** 0.5
            coeff2 = (1 - self.alpha_hat[s]) / (1 - self.alpha[s]) ** 0.5
           
      
            x_s = coeff1 * (x_s - coeff2 * x_pred)

            if s > 0:
                noise = torch.randn_like(x_s)
                sigma = ((1.0 - self.alpha[s - 1]) / (1.0 - self.alpha[s]) * self.beta[s]) ** 0.5
                x_s += sigma * noise     
            
        return x_s
