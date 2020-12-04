from tqdm import tqdm
from src.train import *

params = [
    {'method': 'MyCNN', 'batch_size': 100},
    {'method': 'InceptionV3', 'batch_size': 100}
]

if __name__ == '__main__':

    for i in tqdm(params):
        Train(method=i.get('method'),
              batch_size=i.get('batch_size'))
