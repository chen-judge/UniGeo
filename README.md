# UniGeo

Jiaqi Chen, Tong Li, Jinghui Qin, Pan Lu, Liang Lin, Chongyu Chen, Xiaodan Liang. 
"UniGeo: Unifying Geometry Logical Reasoning via Reformulating Mathematical Expression".
Conference on Empirical Methods in Natural Language Processing (EMNLP 2022)

We construct a large-scale Unified Geometry
problem benchmark, UniGeo, which contains
4,998 calculation problems and 9,543 proving
problems.
We also present a unified multitask
Geometric Transformer framework, Geoformer,
to tackle calculation and proving problems
simultaneously in the form of sequence
generation, which finally shows the reasoning
ability can be improved on both two tasks by
unifying formulation.

## Datasets
Download the UniGeo dataset from [Google Drive](https://drive.google.com/drive/folders/1NifdHLJe5U08u2Zb1sWL6C-8krpV2z2O?usp=share_link).

Create a path *datasets* and move the UniGeo into it.

The code structure is shown below: 

```bash
./datasets
    UniGeo/ 
        proving_test.pk  
        proving_train.pk  
        proving_val.pk  
        ...
        
./Geoformer
    scripts/
    snap/
    src/
    ...
```

## Setup
```bash
# Create python environment
conda create -n unigeo python=3.7
source activate unigeo

# Install python dependencies
pip install -r requirements.txt

# Download T5 backbone checkpoint
python download_backbones.py
```





## Unified Training
Execute this script to train the model.

```bash
cd Geoformer
bash scripts/train.sh 1
```

The pretrained checkpoint can be founded here ([pretrained.pth](https://drive.google.com/drive/folders/1NifdHLJe5U08u2Zb1sWL6C-8krpV2z2O?usp=share_link)).
You can modify the following argument to change the path to pre-trained model.
```bash
--load snap/pretrained
```

## Pre-training
You can also execute this script to pre-train a new model.
```bash
cd Geoformer
bash scripts/pretrain.sh 1
```


## Evaluation
Execute this script to evaluate the model.
```bash
cd Geoformer
bash scripts/evaluate.sh 1
```

The model checkpoint of the reported **Geoformer + Pretraining** can be founded here ([geoformer.pth](https://drive.google.com/drive/folders/1NifdHLJe5U08u2Zb1sWL6C-8krpV2z2O?usp=share_link)).
You can modify the following argument to test *geoformer.pth* or your trained model.
```bash
--load snap/geoformer
```

