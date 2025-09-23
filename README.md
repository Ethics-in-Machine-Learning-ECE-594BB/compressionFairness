# Repository Overview

This repository is a comprehensive collection of code, models, and scripts focused on fairness and compression in machine learning. It contains implementations and experiments on advanced techniques such as adversarial debiasing, fairness constraints, model pruning, quantization, and knowledge distillation.

## Directory Structure

- **data/**  
  Contains raw data files (e.g., `adult.csv`, `adult_test.csv`, `compas.csv`) used for training and evaluation.

- **models/**  
  Stores pre-trained models, including baseline, debiased, pruned, quantized, and distilled variants.

- **scripts/**  
  Contains training, evaluation, pruning, quantization, and fine-tuning scripts. For example, `scripts/basicFCN/train.py` and `scripts/basicFCN/train_fairness.py` offer different training strategies.

- **src/**  
  Provides core logic, model definitions, and data loaders (e.g., `src/simpleFCNN.py` and `src/adultIncome.py`).

- **notebooks/**  
  Contains Jupyter notebooks for experimentation, visualization, and analysis.

- **LICENSE**  
  Licensed under the MIT License.

- **CONVENTIONS.md**  
  Documents repository-specific guidelines and conventions.

## Getting Started

1. **Installation**  
   Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. **Data Preparation**  
   Place your raw data in the `data/raw/` directory (e.g., `adult.csv`, `adult_test.csv`, `compas.csv`).

3. **Training**  
   - For baseline training:
     ```bash
     python scripts/basicFCN/train.py --dataset adult
     ```
   - For adversarial debiasing:
     ```bash
     python scripts/basicFCN/train_fairness.py --dataset adult --debiasing adversarial --protected_attr sex --adv_weight 0.5
     ```
   - For training with fairness constraints:
     ```bash
     python scripts/basicFCN/train_fairness.py --dataset adult --debiasing fairness_constraint --protected_attr sex --fairness_type demographic_parity --lambda_fair 0.3
     ```

4. **Evaluation**  
   Use the provided scripts and notebooks to evaluate model performance and fairness metrics.

## Contributing

Please review the guidelines in `CONVENTIONS.md` before contributing to ensure consistency with the project's conventions.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
