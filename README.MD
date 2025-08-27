# STDiffusion (ICDM 2025)
Official implementation for STDiffusion, published in ICDM 2025. 
## Authors
Zixuan Ma, Chenfeng Huang

## System Specifications (Tested Environment)
* OS: Ubuntu 24.04
* CPU: AMD Ryzen 9 9950X3D (16 cores @ 5.70GHz)
* GPU: NVIDIA GeForce RTX 4090 (24GB VRAM)
* CUDA Version: 11.8 or higher

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/mobkageyama/STDiffusion.git
cd STDiffusion
```

### 2. Environment Setup
```bash
# Create and activate conda environment
conda create -n STDiffusion python=3.8
conda activate STDiffusion

# Install dependencies
pip install -r requirements.txt
```

## Model Training and Sampling

### Training
Start training using the provided script:
```bash
sh train.sh
```

### Sampling
Generate samples using:
```bash
sh sample.sh
```

### Command Format
```bash
python models/trainer.py \
    --config <configuration_filename> \
    --seed <your_luck_number> \
    --note <custom_label> \
    --action <train or sample>
```

### Command Parameters
- `config`: Configuration file path (e.g., `configs/ETTh1.yaml`)
- `seed`: Random seed for reproducibility
- `note`: Custom label for the experiment
- `action`: Choose between 'train' or 'sample'

### Output Directories
- Training checkpoints: `./STDiffusion/saved_models/`
- Generated samples: `./STDiffusion/generated_datasets/`
