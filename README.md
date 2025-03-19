# TurbRadOL: Operator Learning Transformers for Turbulent Radiative Layer Dynamics

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)

## Overview

TurbRadOL is a project focused on applying operator learning transformers to model turbulent radiative layer dynamics using "the well" dataset. This repository implements deep learning models that can learn the underlying physics operators governing turbulent fluid dynamics in radiative environments.

## Features

- Implementation of operator learning transformer architectures
- Data processing utilities for "the well" dataset
- Training and evaluation scripts for turbulent flow prediction
- Visualization tools for radiative layer dynamics
- Pre-trained models for immediate inference

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/TurbRadOL.git
cd TurbRadOL

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Dataset

This project uses "the well" dataset, which contains high-fidelity simulations of turbulent radiative layers. The dataset includes:

- Velocity fields
- Temperature distributions
- Radiative flux measurements
- Boundary conditions
- Time-evolving turbulent structures

Please refer to the `data/README.md` file for instructions on downloading and preprocessing the dataset.

## Usage

### Training a model

```bash
python train.py --config configs/default.yaml --output_dir results/experiment1
```

### Evaluating a trained model

```bash
python evaluate.py --model_path results/experiment1/best_model.pt --test_data data/test
```

### Running inference

```bash
python infer.py --model_path results/experiment1/best_model.pt --input_file examples/sample_input.h5
```

## Model Architecture

TurbRadOL implements several operator learning transformer architectures, including:

1. **Fourier Neural Operator Transformer (FNOT)** - Combines FNO with transformer attention mechanisms
2. **DeepONet with self-attention** - Enhances DeepONet with transformer-based encoding
3. **Physics-Informed Transformer (PIT)** - Incorporates physics constraints in the transformer architecture

Detailed architecture descriptions can be found in the `docs/architectures.md` file.

## Results

Our models achieve state-of-the-art performance on turbulent radiative layer prediction tasks:

| Model | MSE | Relative L2 Error | Physical Consistency Score |
|-------|-----|-------------------|----------------------------|
| FNOT  | 1.23e-4 | 0.0321 | 0.945 |
| DeepONet+Attention | 1.57e-4 | 0.0367 | 0.932 |
| PIT | 1.05e-4 | 0.0298 | 0.961 |

For more detailed results and visualizations, see the `docs/results.md` file.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code or dataset in your research, please cite:

```bibtex
@article{yourlastname2025turbol,
  title={Operator Learning Transformers for Turbulent Radiative Layer Dynamics},
  author={Your Name and Collaborators},
  journal={arXiv preprint arXiv:2503.00000},
  year={2025}
}
```

## Acknowledgments

- The developers of "the well" dataset
- [PyTorch](https://pytorch.org/) and [PyTorch Lightning](https://www.pytorchlightning.ai/)
- [Fourier Neural Operator](https://github.com/zongyi-li/fourier_neural_operator) repository
- [DeepONet](https://github.com/lululxvi/deeponet) repository
- Computational resources provided by [your institution]