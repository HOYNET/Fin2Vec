import yaml


with open('config.yaml') as f:
    y = yaml.load(f,Loader=yaml.FullLoader)

print(y["HoynetConfig"])