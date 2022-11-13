

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

