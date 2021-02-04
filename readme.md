# Robust Embedding Learning
This repository contains the code for experiments with learning robust embeddings.

## Prepare environment

``` shell
conda env create -n <myenv> -f environment.yaml
conda activate <myenv>
```

## Evaluation
Generally this repository follow most of the guidelines presented by [Roth et al. (2020)](https://arxiv.org/abs/2002.08473), with a few exceptions:

1. The learning rate stated in the paper is not equal to the actual learning rate used throughout experiments, we use the one from the experiments (`1e-5` and not `10e-5`).
2. They use the default crop sizes of `torchvision.transforms.RandomResizedCrop` (which select patches between 8%-100% of images). This lower bound seems counter-intuitively low and stem from [Szegedy et al. (2014)](https://arxiv.org/abs/1409.4842) (which I deem outdated).
3. They split train-test purely based on sorting labels, not accounting for class variance within the two sets.
4. They replay their experiment across multiple random seeds, this is (normally done) to account for initialization variances, however, they use transfer learning so there is no initialization. Thereby the seed only reflect the sampling for batches, which is negligible compared to 3. Thereby, we don't do this (currently).


# Experiments
## Training
### Natural
 - Losses (Contrastive, Triplet)
 - Datasets (CUB, SOP, CARS196)
 => 6 runs

### Evaluations
 - Models (6 variants)
 - Perturbation Spaces (No attack, Linf, L2, L0)
 - Attacks (PGD, CW, FGSM)

 <!-- - Perturbations Spaces (Linf, L2, L0) -->
 <!-- - Attack methods (PGD, CW, FGSM) -->
 -
