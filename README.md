# Сomment analyzer

## Training of NER Extractor for your own data

###### Project was run on Ubuntu 20.04 and Python3.8

## How to setting

 - Install driver for your gpu:
 - Install [cuda toolkit](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_network)
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
git clone this repository
```
- Install project requirements:
```
cd ner_training_on_transformers
pip install -r requirements.txt
```

## How to start

- To see the project in graphical view enter:
```
kedro viz         
```
