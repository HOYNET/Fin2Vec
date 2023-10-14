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
* sample configurations(.yml) are in `demo/training` directory

### PCRN training (logging)
  ```bash
  $ python ./pcrn.py -c ./demo/training/pcrn.yml >> log.txt
  ```
### Fin2Vec training (logging)
  ```bash
  $ python ./fin2vec.py -c ./demo/training/fin2vec.yml >> log.txt
  ```

## Demo
* check configuration(.yml) carefully and customize it
* sample configurations(.yml) are in `demo/inference` or `demo/wordClustering` directory
  
### Fin2Vec Inference
  ```bash
  $ python ./inference.py -c ./demo/inference/inference.yml
  ```
### Clustering with Word Embedding
  ```bash
  $ python ./wordClustering.py -c ./demo/wordClustering/wordClustering.yml
  ```

## Visualize

### Visualize Loss
  ```bash
  $ python visualize/lossVisualize.py -f log.txt -i Title -l True
  ```
