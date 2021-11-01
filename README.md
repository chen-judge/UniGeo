
# UniGeo



## Environment
python=3.6

allennlp==0.9.0

pip install requirements.txt

Document for <a href="http://docs.allennlp.org/v0.9.0/index.html">allennlp</a>


## Usage 

### Pretraining

    allennlp train config/NGS_pretrain_Eng.json --include-package NGS_pretraining -s save/test

### Training
    
    allennlp train config/NGS_Aux_Eng.json --include-package NGS_Aux -s save/test
