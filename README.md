
# DS6050_G12_PROJECT

Repository for uvaMSDS 6050 Group 12 Project

From Baseline Models to Deep Networks: A
Systematic Comparison of Model Complexity and
Training Strategies for Fashion Image Classification

This project investigates image classification on the
FashionMNIST dataset using a comparative modeling approach.
This project develops a complete pipeline for supervised single
label classification, beginning with a traditional non-deep learn-
ing method, the XGBoost and extending to Multilayer Percep-
tron (MLP) model as baselines to more advanced architectures
including a simple Convolutional Neural Network (CNN) with
2 layers, and a deeper CNN of 3 convulsions with enhanced
feature extraction capacity. The impacts of architectural depth,
regularization strategies, and hyperparameter optimization on
model performances are evaluated. Additionally, the effects
of data normalization, training, validation splitting strategies,
and augmentation techniques on generalization are explored.
These comparative experiments highlight the trade offs between
classical machine learning methods and modern deep learning
architectures for image classification.


## Running on UVA Rivanna

Clone the project

```bash
  git clone https://github.com/d26clarke/DS6050_G12_PROJECT.git
```

Go to the project directory

```bash
  cd DS6050_G12_PROJECT
```

Install dependencies

```bash
  bash scripts/setup_env.sh
```

## Usage/Examples

Create ablations

```bash
  python scripts/generate_ablation_configs.py
```

Run ablations

```bash
  chmod +x scripts/launch_ablations.sh
  ./scripts/launch_ablations.sh <your selected environment: dev | sit | prod >
```

Run single model
```bash
  sbatch slurm/run_single.slurm configs/{YOUR_SELECTED_MODEL}.yaml <your selected environment: dev | sit | prod >
```



## Screenshots

Per Class F1 Scores

![App Screenshot](https://github.com/d26clarke/DS6050_G12_PROJECT/blob/main/media/per-class%20f1%20score.png)

