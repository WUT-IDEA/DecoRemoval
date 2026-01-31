# DecoRemoval: Factor Decorrelation Enhanced Data Removal from Deep Predictive Models

> 🔐 Official implementation of **DecoRemoval**, a robust and efficient certified unlearning method designed for deep predictive models, especially under out-of-distribution (OOD) settings.  
> 📝 Accompanies the NeurIPS 2025 paper: *Factor Decorrelation Enhanced Data Removal from Deep Predictive Models*

## 🌟 Highlights

- **Retraining-Free Data Unlearning**: Approximates full retraining via certified Newton update.
- **Feature Decorrelation via RFF**: Uses Random Fourier Features for dimensionality reduction while preserving discriminative structure.
- **Smoothed Removal with Loss Perturbation**: Prevents gradient leakage and improves robustness.
- **Strong OOD Performance**: Outperforms baselines in both accuracy and efficiency under distribution shift.
- **Broad Applicability**: Supports vision, text, and structured data (MNIST, CIFAR-10, SST-2, ESS, CGSS).

## 🛠 Installation

```bash
git clone https://github.com/your-org/DecoRemoval.git
cd DecoRemoval
conda create -n decoremoval python=3.9
conda activate decoremoval
pip install -r requirements.txt
```

## 📁 Dataset Setup

You can download and preprocess the datasets using the helper script:

```bash
python scripts/download_and_preprocess.py --dataset MNIST
# Available datasets: MNIST, CIFAR10, SST2, ESS, CGSS
```

## 🚀 Running Experiments

### Step 1: Train the Base Model

```bash
python train_model.py --dataset MNIST --backbone MLP
```

### Step 2: Remove Data with DecoRemoval

```bash
python remove_data.py --dataset MNIST --method DecoRemoval --remove_size 1000
```

### Step 3: Evaluate

```bash
python evaluate.py --dataset MNIST --method DecoRemoval
```

### 🎯 Baselines Included

- Certified Removal (CR)  
- SISA (Sharded Training)  
- DP-SGD (Differential Privacy)  
- Certified Unlearning (CU, 2024)  
- SSD (Selective Synaptic Dampening, 2024)

### 📊 Accuracy (ACC%) and F1 Score Comparison across Methods and Removal Sizes

> The closer to **Retrain**, the better. Bold = Best, _Underline_ = Second Best

| Dataset   | Samples | Retrain ACC | Retrain F1 | CR ACC | CR F1  | SISA ACC | SISA F1  | DP-SGD ACC | DP-SGD F1  | SSD ACC | SSD F1  | CU ACC | CU F1  | DR (Ours) ACC | DR F1 |
|-----------|---------|-------------|-----|--------|-----|----------|-----|-------------|-----|----------|-----|--------|-----|----------------|-----|
| **MNIST** | 1000    | 51.75       | 0.505 | 43.13 | 0.394 | 43.64   | 0.405 | 45.78     | 0.440 | 45.45   | 0.458 | _47.35_ | _0.458_ | **48.97**     | **0.482** |
|           | 3000    | 51.35       | 0.498 | 42.21 | 0.391 | 43.46   | 0.401 | 45.34     | 0.438 | 45.24   | 0.455 | _46.94_ | _0.455_ | **48.65**     | **0.478** |
|           | 10000   | 51.02       | 0.495 | 41.87 | 0.390 | 42.96   | 0.398 | 44.87     | 0.432 | 45.03   | 0.450 | _46.53_ | _0.450_ | **48.34**     | **0.473** 
| **CIFAR-10** | 1000  | 50.76       | 0.501 | 43.09 | 0.392 | 43.21   | 0.401 | 45.30     | 0.436 | 45.06   | 0.452 | _46.84_ | _0.452_ | **48.56**     | **0.478** |
|              | 3000  | 50.46       | 0.496 | 42.29 | 0.391 | 42.94   | 0.396 | 44.84     | 0.433 | 44.72   | 0.448 | _46.52_ | _0.449_ | **48.14**     | **0.473** |
|              | 10000 | 50.01       | 0.491 | 41.76 | 0.389 | 42.51   | 0.392 | 44.37     | 0.428 | 44.51   | 0.443 | _46.11_ | _0.444_ | **47.83**     | **0.469** 
| **SST-2** | 1000    | 91.76       | 0.843 | 89.71 | 0.808 | 89.98   | 0.817 | _90.45_   | _0.825_ | 89.98 | 0.818 | 89.94 | 0.813 | **90.45**     | **0.827** |
|           | 3000    | 91.55       | 0.840 | 89.65 | 0.801 | 89.76   | 0.814 | _90.36_   | _0.820_ | 89.86 | 0.816 | 89.77 | 0.808 | **90.39**     | **0.825** |
|           | 10000   | 91.14       | 0.839 | 89.48 | 0.796 | 89.65   | 0.809 | _90.10_   | _0.816_ | 89.87 | 0.816 | 89.67 | 0.805 | **90.38**     | **0.821** 
| **ESS**   | 1000    | 55.43       | 0.540 | 48.61 | 0.410 | 48.64   | 0.420 | 50.47     | 0.450 | _50.15_ | _0.486_ | 51.34 | 0.490 | **54.97**     | **0.520** |
|           | 3000    | 55.35       | 0.540 | 48.41 | 0.400 | 48.46   | 0.410 | 50.01     | 0.440 | _50.01_ | _0.479_ | 51.15 | 0.480 | **54.85**     | **0.510** |
|           | 10000   | 55.24       | 0.530 | 48.40 | 0.390 | 48.44   | 0.410 | 49.67     | 0.430 | _49.19_ | _0.478_ | 50.87 | 0.470 | **54.64**     | **0.510** |
| **CGSS**  | 1000    | 51.60       | 0.515 | 41.24 | 0.465 | 43.52   | 0.472 | 46.76     | _0.487_ | _47.77_ | 0.475 | 48.77 | 0.485 | **50.82**     | **0.501** |
|           | 3000    | 51.32       | 0.506 | 40.74 | 0.458 | 43.02   | 0.469 | 46.25     | _0.482_ | _47.31_ | 0.474 | 48.21 | 0.480 | **50.48**     | **0.496** |
|           | 10000   | 50.98       | 0.498 | 40.15 | 0.434 | 42.75   | 0.465 | 45.95     | _0.473_ | _47.15_ | 0.471 | 47.85 | 0.477 | **50.10**     | **0.495** |

## 📦 Code Structure

```
.
├── core/                # Core algorithms: RFF mapping, weight optimization, Newton update
├── models/              # Model architectures (MLP, LSTM, Transformer)
├── datasets/            # Data loading and preprocessing
├── scripts/             # Helper scripts for training and evaluation
├── experiments/         # Configs and logs
├── main.py              # Entry point for training/removal
└── README.md
```

## 🧠 Citation

If you use this work in your research, please cite:

```bibtex
@article{anonymous2025decoremoval,
  title={Factor Decorrelation Enhanced Data Removal From Deep Predictive Models},
  author={Anonymous},
  journal={NeurIPS},
  year={2025}
}
