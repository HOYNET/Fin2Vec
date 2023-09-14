## Prerequisites
* install conda
* create a new conda environment with requirements.txt
```bash
        conda create --name [your_enviromnet_name] --file requirements.txt
```
* activate the conda environment
```bash
        conda activate [your_environment_name]
```
* deactivate the conda enviroment when finished the jobs
```bash
        conda deactivate
```

* training
  ```bash
  python training.py -p ./data/NASDAQ_DT_FC_STK_QUT.csv -c ./data/NASDAQ_FC_STK_IEM_IFO.csv -b 100 -e 100 -l 0.0001
  ```
