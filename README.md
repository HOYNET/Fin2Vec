# Fin2Vec
## Prerequisites
### install conda
* create a new conda environment named hoynet with environment.yml
* PyTorch is essential. Set PyTorch according to your own local environment
```bash
$ conda env create -f environment.yml
```
### activate the conda environment
```bash
$ conda activate hoynet
```
### deactivate the conda enviroment when finished the jobs
```bash
$ conda deactivate
```
## Training
* check configuration(.yml) carefully and customize it
### PCRN training
  ```bash
  $ python ./pcrn.py -c ./demo/training/pcrn.yml
  ```
### Fin2Vec training
  ```bash
  $ python ./fin2vec.py -c ./demo/training/fin2vec.yml
  ```
## Demo
* check configuration(.yml) carefully and customize it
### Fin2Vec Inference
  ```bash
  $ python ./inference.py -c ./demo/inference/inference.yml
  ```
### Clustering with Word Embedding
  ```bash
  $ python ./wordClustering.py -c ./demo/wordClustering/wordClustering.yml
  ``` 
