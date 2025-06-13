from STDiffusion import DiffModel
from torch.optim import Adam
import argparse
import torch
import datetime
import yaml
import os
from data_loader import get_csv_scaled_dataloader
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
import numpy as np


class Trainer(object):
    def __init__(self, config, model, dataloader, device, model_name, seed=123):
        super().__init__()
        self.device = device
        self.config = config
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'saved_models/'
        )
        if not os.path.exists(path):
            os.makedirs(path)
        self.model_path = os.path.join(path,f'{model_name}.pt')

        self.dataloader = dataloader
        self.model = model
        self.model_name = model_name

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            if device == 'cuda':
                torch.cuda.manual_seed_all(seed)

    def train(self):
        torch.cuda.empty_cache()
        start_time = datetime.datetime.now()
        optimizer = Adam(self.model.parameters(), lr=self.config['lr'], weight_decay=1e-6)

        p1 = int(0.75 * self.config['epochs'])
        p2 = int(0.9 * self.config['epochs'])
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[p1, p2], gamma=0.1)
        for epoch_no in tqdm(range(self.config['epochs']), desc="Training"):  
            avg_loss = 0
            batch_count = 0
            self.model.train()
            with tqdm(self.dataloader) as it:
                for batch_no, train_data in enumerate(it, start=1):
                    optimizer.zero_grad()                
                    loss = self.model(train_data.to(self.device))
                    loss.backward()
                    clip_grad_norm_(self.model.parameters(), 1.0)

                    for name, param in self.model.named_parameters():
                        if param.grad is None:
                            print(f'Epoch {epoch_no}, {name} grad: {param.grad}')
                            import ipdb; ipdb.set_trace()

                    avg_loss += loss.item()
                    batch_count += 1
                    optimizer.step()
                    it.set_postfix(
                        ordered_dict={
                            'avg_epoch_loss': avg_loss / batch_no,
                            'epoch': epoch_no,
                        },
                        refresh=False,
                    )
                
                lr_scheduler.step()


 
        torch.save({
            'model_state_dict': self.model.state_dict(),
        }, self.model_path)

        end_time = datetime.datetime.now()
        duration = end_time - start_time
        print(f'Training completed in: {duration}')

    def sample(self):
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        start_time = datetime.datetime.now()
        self.model.eval()
        samples = self.model.generate()
        end_time = datetime.datetime.now()
        duration = end_time - start_time
        print(f'Generation completed in: {duration}')

       
        sample_path =  os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        f'generated_datasets/{self.model_name}' 
        )
        np.save(sample_path, samples.cpu().numpy())



        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='STDiffusion')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--note', type=str, default='')
    parser.add_argument('--action', type=str, required=True, help='train or sample')
    args = parser.parse_args()
    print(args)

    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config/')
    path = path + args.config + '.yaml'
    with open(path, 'r') as f:
        config = yaml.safe_load(f)

    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Using device: {device}')


    config_train = config['train']
    dataset = config['dataset']


    full_loader, train_loader, test_loader, dataset_info = get_csv_scaled_dataloader(dataset, 
    seq_len=config_train['seq_len'], batch_size=config_train['batch_size'])

    model = DiffModel(dataset_info, config, device).to(device)
    model_name = f'model_{dataset}_{args.note}'
    
    if args.action == 'train':
        trainer = Trainer(config_train, model, full_loader, device, model_name, args.seed)
        trainer.train()

    elif args.action == 'sample':
        trainer = Trainer(config_train, model, full_loader, device, model_name, args.seed)
        trainer.sample()

    else:
        raise NotImplementedError 
 