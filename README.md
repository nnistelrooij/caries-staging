# Staging caries around restorations in bitewing radiographs

This is the code repository accompanying the paper *"Niels van Nistelrooij, Eduardo Chaves, et al., 2024. Deep learning-based algorithm for staging caries around restorations in bitewings"* submitted to Caries Research.


## Installation

Please refer to `setup.sh` for the installation of a virtual environment and the necessary packages.


## Reproduction

### Pre-processing

With the bitewing radiographs and annotations from multiple annotators as supplied on OSF, the data can be pre-processed and split for 10-fold cross-validation by running

``` shell
PYTHONPATH=code python code/caries/preprocess.py
```

### Two-stage training

The algorithm for detecting and staging caries lesions around restorations uses a two-stage training approach. The first stage detects teeth, restorations, residual caries lesions, and primary/secondary caries lesions. The second stage detects primary and secondary caries lesions and predicts a lesion severity score from 0 to 1.

#### Stage 1

Specify the cross-validation split you would like to train within `caries/configs/stage1.py` and run the following in the terminal:

``` shell
PYTHONPATH=code python code/mmdetection/tools/train.py code/caries/configs/stage1.py
```

While training is running, several metrics are logged to TensorBoard in the working directory specified by `work_dir` in the configuration file.

#### Stage 2

Likewise, specify the cross-validation split you would like within `caries/configs/stage2.py` after training the first stage for that split and run the following in the terminal:

``` shell
PYTHONPATH=code python code/mmdetection/tools/train.py code/caries/configs/stage2.py
```

The model parameters will be initialized based on the stage 1 model to fine-tune the effectiveness for detecting and staging primary and secondary caries lesions.


### Inference

Choose the latest or best checkpoint from the working directory and run the following to save the annotations and predictions to a pickle file in the working directory.

```shell
PYTHONPATH=code python code/mmdetection/tools/test.py code/caries/configs/stage2.py work_dirs/stage2_<split>/<checkpoint>.pth --tta [--show [--wait-time <seconds>]] 
```

### Evaluation

The annotations can be compared to the predictions visually, by specifying `--show` and `--wait-time` after the inference terminal command above.

## Citation

```
@article{10.1159/000542289,
    author = {van Nistelrooij, Niels and Chaves, Eduardo Trota and Cenci, Maximiliano Sergio and Cao, Lingyun and Loomans, Bas A.C. and Xi, Tong and El Ghoul, Khalid and Romero, Vitor Henrique Digmayer and Lima, Giana Silveira and Flügge, Tabea and van Ginneken, Bram and Huysmans, Marie-Charlotte and Vinayahalingam, Shankeeth and Mendes, Fausto Medeiros},
    title = {Deep Learning-Based Algorithm for Staging Secondary Caries in Bitewings},
    journal = {Caries Research},
    pages = {1-11},
    year = {2024},
    month = {10},
    doi = {10.1159/000542289}
}
```

Additionally, confusion matrices and ROC curves can be made by running `caries/evaluation/tooth_level.py`. Furthermore, the predictions of lesion severity scores can be evaluated by running `caries/evaluations/lesion_level.py`.
