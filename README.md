# Protein-Angles-Prediction

## Quick Summary
Deep Learning code uses PyTorch.

The size of the training, validation, and test set is 9530, 926, and 245 proteins (respectively). The proteins were culled from the PISCES server with constraints such as high resolution (<2.5Å) and sequence identity cutoff of 25%. The DSSP program was used to calculate "true" angles from PDB files. The data files are located in the `data` folder.

Trained models are located in the `model` folder.

| Model  | Input Features | PHI (Dev MAE) | PSI (Dev MAE) | PHI (Test MAE) | PSI (Test MAE)
| ------------- | ------------- | :-------------: | :-------------: | :-------------: | :-------------: |
| Residual CNN  | Simple one-hot vectors denoting amino acid types  | 24.47 | 47.19 | 24.63 | 48.80 |
| LSTM-BRNN  | Simple one-hot vectors denoting amino acid types  | 24.17 | 44.61 | 23.96 | 44.84  |
| Residual CNN | 7 physio-chemical properties + 3-state secondary structures predicted by PSIPRED | 19.78 | 30.95 | 20.27 | 33.06 | 
| Residual CNN | 7 physio-chemical properties + 3-state secondary structures predicted by PSIPRED + Prediction by SOLVPRED |  19.24 | 29.73 | 19.78 | 32.06 | 
| Residual CNN	| 7 physio-chemical properties + 3-state secondary structures predicted by PSIPRED + Prediction by SOLVPRED + PSSM | 18.54 | 28.36 | 19.06 | 30.57 | 
| LSTM-BRNN	| 7 physio-chemical properties + 3-state secondary structures predicted by PSIPRED + Prediction by SOLVPRED + PSSM |  18.14 | 27.51 | 18.74 | 29.55 |

## Training
Run the command `python train.py` to start training a LSTM-BRNN model. Refer to the code for parsing arguments at the beginning of `train.py` for more command options. Few examples:

1. Run the following command to train a Residual-CNN model: `python train.py --model_type residual_cnn`

2. Run the following command to use input features different from one-hot vectors: `python train.py --input_types BLOSUM PHYS`


## Related Papers
LSTM-BRNN is used in the following papers:
* *SPIDER3*: [Capturing non-local interactions by long short-term memory bidirectional recurrent neural networks for improving prediction of protein secondary structure, backbone angles, contact numbers and solvent accessibility](https://www.ncbi.nlm.nih.gov/pubmed/28430949)
* *SPIDER3-Single*: [Single-sequence-based prediction of protein secondary structures and solvent accessibility by deep whole-sequence learning](https://www.ncbi.nlm.nih.gov/pubmed/30368831)

## [WIP]

Observation: If we know the true secondary structure of a protein, we can predict the backbone torsion angles very accurately. The performance of `python3 train.py --input_types ONEHOT TRUE_SS --model_type residual_cnn` on dev dataset is at least better than [16.15, 18.16].

