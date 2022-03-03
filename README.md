# HH-VAEM

This repository contains the official Pytorch implementation of the Hierarchical Hamiltonian VAE (**HH-VAEM**) model and the sampling-based feature acquisition technique presented in the paper 
[Missing Data Imputation and Acquisition with Deep Hierarchical Models and Hamiltonian Monte Carlo](https://arxiv.org/pdf/2202.04599.pdf).  HH-VAEM is a Hierarchical VAE model for mixed-type incomplete data that uses Hamiltonian Monte Carlo with automatic hyper-parameter tuning for improved approximate inference. The repository contains the implementation and the experiments provided in the paper.
<br>
<p align="center">
  <img width="300" src="imgs/hh-vaem.png">
</p>
<br>

Please, if you use this code, cite the [preprint](https://arxiv.org/pdf/2202.04599.pdf) using:
```
@article{peis2022missing,
  title={Missing Data Imputation and Acquisition with Deep Hierarchical Models and Hamiltonian Monte Carlo},
  author={Peis, Ignacio and Ma, Chao and Hern{\'a}ndez-Lobato, Jos{\'e} Miguel},
  journal={arXiv preprint arXiv:2202.04599},
  year={2022}
}
```

## Instalation 
The installation is straightforward by creating a conda virtual environment named <code>HH-VAEM</code>, using the provided <code>environment.yml</code> file:
```
conda env create -f environment.yml
```

## Usage

### Training
The project is developed in the recent research framework [PyTorch Lightning](https://www.pytorchlightning.ai/). The HH-VAEM model is implemented as a [<code>LightningModule</code>](https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html) that is trained by means of a [<code>Trainer</code>](https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html). A model can be trained by using:
```
# Example for training HH-VAEM on Boston dataset
python train.py --model HHVAEM --dataset boston --split 0
```
This will automatically download the <code>boston</code> dataset, split in 10 train/test splits and train HH-VAEM on the training split <code>0</code>. Both <code>data/</code> and <code>logs/</code> folders will be created to store the dataset and model checkpoints, respectively. The variable <code>LOGDIR</code> can be modified in <code>src/configs.py</code> to change the directory where these folders will be created.

The following datasets are available:
- A total of 10 UCI datasets: <code>avocado</code>, <code>boston</code>, <code>energy</code>, <code>wine</code>, <code>diabetes</code>, <code>concrete</code>, <code>naval</code>, <code>yatch</code>, <code>bank</code> or <code>insurance</code>.
- The MNIST datasets: <code>mnist</code> or <code>fashion_mnist</code>.

And also the following models (implemented in <code>src/models/</code>):
- <code>HHVAEM</code>: the proposed model in the paper.
- <code>VAEM</code>: the VAEM strategy presented in [(Ma et al., 2020)](https://arxiv.org/pdf/2006.11941.pdf) with Gaussian encoder (without including the
Partial VAE).
- <code>HVAEM</code>: A Hierarchical VAEM with two layers of latent variables and a Gaussian encoder.
- <code>HMCVAEM</code>: A VAEM that includes a tuned HMC sampler for the true posterior.
- For MNIST datasets (non heterogeneous data), use <code>HHVAE</code>, <code>VAE</code>, <code>HVAE</code> and <code>HMCVAE</code>.

By default, the test stage will be executed at the end of the training stage. This can be cancelled with <code>--test 0</code> for manually running the test using:
```
# Example for testing HH-VAEM on Boston dataset
python test.py --model HHVAEM --dataset boston --split 0
```
which will load the trained model to be tested on the <code>boston</code> test split number <code>0</code>. Once all the splits are tested, the average results can be obtained using the script in the <code>run/</code> folder:
```
# Example for obtaining the average test results with HH-VAEM on Boston dataset
python test_splits.py --model HHVAEM --dataset boston
```
### Experiments
The experiments in the paper can be executed using:
```
# Example for running the SAIA experiment with HH-VAEM on Boston dataset
python active_learning.py --model HHVAEM --dataset boston --method mi --split 0

# Example for running the OoD experiment using MNIST and Fashion-MNIST as OoD:
python ood.py --model HHVAEM --dataset mnist --dataset_ood fashion_mnist --split 0
```
Once this is executed on all the splits, you can plot the SAIA error curves or obtain the average OoD metrics using the scripts in the <code>run/</code> folder:
```
# Example for running the SAIA experiment with HH-VAEM on Boston dataset
python active_learning_plots.py --models VAEM HHVAEM --dataset boston

# Example for running the OoD experiment using MNIST and Fashion-MNIST as OoD:
python ood.py --model HHVAEM --dataset mnist --dataset_ood fashion_mnist --split 0
```

## Contributors
[Ignacio Peis](http://www.tsc.uc3m.es/~ipeis/index.html) <br>
[Chao Ma](https://chao-ma.org/) <br>
[José Miguel Hernández-Lobato](https://jmhl.org/) <br>

## Contact
For further information: <a href="mailto:ipeis@tsc.uc3m.es">ipeis@tsc.uc3m.es</a>
