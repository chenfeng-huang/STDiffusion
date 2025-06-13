# STDiffusion (In Submission)

## Authors
Anonymous for Triple-blind reviewing

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


# Longterm sequence generation result details
Performance comparison of long sequence time series generation methods on ETTh1, ETTh2 and Exchange datasets. Lower scores indicate better performance across all metrics.


## Discriminative Score (Lower = Better)
*Measures how well the model generates realistic time series data*

| Method | ETTh1-64 | ETTh1-128 | ETTh1-256 | ETTh2-64 | ETTh2-128 | ETTh2-256 | Exchange-64 | Exchange-128 | Exchange-256 |
|--------|-----------|-----------|-----------|-----------|------------|------------|-------------|--------------|--------------|
| STDiffusion | 0.061±.011 | 0.077±.015 | 0.077±.027| 0.025±.011| 0.038±.014| 0.038±.015| 0.014±.008| 0.036±.020|0.051±.009 |
| Diffusion-TS |0.106±.048 |0.144±.060 |0.060±.030 | 0.047±.011| 0.057±.005|0.079±.010 |0.024±.010 |0.066±.019 | 0.041±.018|
| TimeGAN |0.227±.078 |0.188±.074 |0.442±.056 | 0.215±.023| 0.175±.054| 0.415±.074| 0.220±.023| 0.180±.034| 0.428±.063|
| TimeVAE | 0.171±.142|0.154±.087 | 0.178±.076| 0.165±.123| 0.148±.032| 0.182±.023|0.168±.067 | 0.150±.054| 0.175±.074|
| KoVAE |0.294±.023 | 0.286±.034|0.265±.450 | 0.088±.035|0.142±.034 |0.175±.063 | 0.098±012|0.240±.064 | 0.240±.073|
| Diffwave |0.254±.074  |0.274±.047  |0.304±.068  | 0.462±.051| 0.493±.034| 0.453±.021|0.433±.033 | 0.498±.012|0.486±.046 |
| Cot-GAN |0.296±.048  | 0.451±.080 | 0.461±.010 |0.485±.034| 0.445±.034| 0.445±.064| 0.290±.033| 0.448±.036|0.452±.035 |

---

## Predictive Score (Lower = Better)
*Measures the model's ability to predict future time series values*

| Method | ETTh1-64 | ETTh1-128 | ETTh1-256 | ETTh2-64 | ETTh2-128 | ETTh2-256 | Exchange-64 | Exchange-128 | Exchange-256 |
|--------|-----------|-----------|-----------|-----------|------------|------------|-------------|--------------|--------------|
| STDiffusion | 0.114±.005| 0.116±.004| 0.106±.005|0.100±.003 |0.104±.007|0.107±.018 | 0.034±.001|0.033±.004 | 0.032±.002|
| Diffusion-TS |0.116±.000 | 0.110±.003| 0.109±.013|0.101±.003 |0.110±.006 |0.129±.032 |0.035±.002 |0.035±.001 |0.033±.003 |
| TimeGAN |0.132±.008 | 0.153±.014| 0.220±.008| 0.128±.003| 0.148±.002| 0.215±.002| 0.125±.002| 0.145±.002| 0.185±.003|
| TimeVAE |0.118±.004 | 0.113±.005| 0.110±.027| 0.115±.001| 0.115±.001| 0.112±.007|0.110±.002 | 0.110±.004| 0.095±.009|
| KoVAE |0.124±.002 | 0.122±.004|0.126±.010 |0.116±.004 | 0.117±.003|0.114±.003 |0.036±.001 |0.038±.002 | 0.042±.003|
| Diffwave | 0.133±.008 | 0.129±.003 |0.132±.001  | 0.168±.001|0.161±.001 |0.167±.007 |0.071±.005| 0.082±.004|0.088±.008 |
| Cot-GAN |0.135±.003  |0.126±.001  | 0.129±.000 | 0.140±.007| 0.128±.003| 0.135±.006|0.128±.007|0.125±.005 |0.120±.004 |

---

## Context-FID Score (Lower = Better)
*Measures distributional similarity between generated and real data*

| Method | ETTh1-64 | ETTh1-128 | ETTh1-256 | ETTh2-64 | ETTh2-128 | ETTh2-256 | Exchange-64 | Exchange-128 | Exchange-256 |
|--------|-----------|-----------|-----------|-----------|------------|------------|-------------|--------------|--------------|
| STDiffusion |0.181±.004 |0.206±.011 |0.364±.050 | 0.108±.020|0.183±.030 | 0.244±.018| 0.042±.004| 0.134±.020| 0.208±.036|
| Diffusion-TS | 0.631±.058|0.787±.062 |0.423±.038 |0.117±.020 | 0.201±.032| 0.374±.065| 0.059±.009|0.077±.009 |0.120±.013 |
| TimeGAN | 1.130±.102| 1.553±.169| 5.872±.208 |1.441±.079 | 1.397±.063| 3.288±.123|0.221±.027 |0.452±.018 | 1.023±.032|
| TimeVAE |0.827±.146 |1.062±.134 | 2.826±.093 | 0.522±.058| 1.104±.012| 1.003±.053| 0.161±.019| 0.304±.012| 0.642±.032|
| KoVAE |2.476±.624 |3.513±.119 |4.396±.280 |0.468±.084 |1.307±.049 | 2.235±.035| 0.234±.028|0.318±.014 | 1.833±.034|
| Diffwave |1.543±.153 |2.354±.170 | 2.899±.289|0.602±.108 | 1.603±.096|2.353±.352 |0.301±.036 |1.231±.027 | 1.673±.431|
| Cot-GAN | 3.008±.277 |  2.639±.427| 4.075±.894  | 1.173±.211| 2.675±.107|3.513±.125 | 0.587±.070| 1.259±.030| 1.103±.301|

---

## Correlation Score (Lower = Better)
*Measures distributional similarity between generated and real data*

| Method | ETTh1-64 | ETTh1-128 | ETTh1-256 | ETTh2-64 | ETTh2-128 | ETTh2-256 | Exchange-64 | Exchange-128 | Exchange-256 |
|--------|-----------|-----------|-----------|-----------|------------|------------|-------------|--------------|--------------|
| STDiffusion | 0.033±.006|0.048±.007 | 0.078±.015| 0.059±.008|0.106±.018 |0.091±.010 |0.054±.034 |0.077±.027 |0.059±.049 |
| Diffusion-TS | 0.082±.005| 0.088±.005| 0.064±.005| 0.082±.020| 0.112±.020|0.115±.015 |0.082±.030 | 0.088±.017| 0.082±.022|
| TimeGAN | 0.483±.019|0.188±.006 | 0.576±.006| 0.423±.042| 0.231±.059| 0.763±.009| 0.321±.055| 0.421±.083|0.789±.093 |
| TimeVAE |0.067±.019 | 0.054±.007| 0.046±.006| 0.055±.025|0.108±.053 | 0.167±.003|0.125±.041 |0.204±.080 |0.321±.008 |
| KoVAE |0.103±.007 |0.122±.002 |0.242±.009 | 0.241±.007|0.207±.024 |0.393±.018 | 0.212±.046| 0.163±.047|0.181±.019 |
| Diffwave | 0.186±.008| 0.203±.006| 0.199±.003| 0.204±.005|0.196±.005 | 0.321±.009| 0.312±.031| 0.421±.084|0.732±.082 |
| Cot-GAN |  0.271±.007| 0.176±.006|  0.222±.010| 0.313±.002| 0.204±.0.01| 0.210±.003|0.421±.042 |0.611±.089 |0.891±.012 |

---
  - Values shown as mean ± 95% confidence internval
