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
* <span style="color: red;">Caution! Ensure the configuration is well-written to recognize the device capabilities and prevent forced shutdowns </span>
* <span style="color: red;">주의! 기기 성능에 맞게 실행 구성을 작성하시오. </span>
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
* <span style="color: red;">Caution! Ensure the configuration is well-written to recognize the device capabilities and prevent forced shutdowns </span>
* <span style="color: red;">주의! 기기 성능에 맞게 실행 구성을 작성하시오. </span>
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
