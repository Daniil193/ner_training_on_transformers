## Fine-tutning of NER Extractor for your own data

###### Project was run on Ubuntu 20.04 and Python3.8

## How to setting

 - Install driver for your gpu & [cuda toolkit](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_network)
 - Create virtual env
 ```
 python3 -m venv ner_env
 ```
 - And activate it:
 ```
 source ner_env/bin/activate
 ```
 - Update pip:
 ```
 python3 -m pip install --upgrade pip
 ```
 - Clone repository:
```
git clone https://github.com/Daniil193/ner_training_on_transformers.git
```
- Install project requirements:
```
cd ner_training_on_transformers
pip install -r requirements.txt
```

## How to start

### Prepare data
- You can find more information about example data of this repo [here](https://github.com/dialogue-evaluation/RuNNE)

- For fine-tuning model we need the data splitted on train and valid part. For example, take data as mentioned in [data/1_raw](https://github.com/Daniil193/ner_training_on_transformers/tree/main/data/1_raw), where:
```
_sentences.txt - input raw data
_labels.txt - labels for input raw data
```
- Each sentence should be write in a new line and splitted on tokens with separator, for example "||"
- Also, labels should be writen for each sentence in a new line
- Next, we need to set up config file for preparing data:
```
~/config/dataload.yaml

files:
  p_train_tokens: data/1_raw/train_sentences.txt
  .
  .
```
