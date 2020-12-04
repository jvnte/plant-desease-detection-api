from tqdm import tqdm
from src.train import *

params = [
    {'method': 'my_cnn', 'batch_size': 100},
    {'method': 'inception_v3', 'batch_size': 100}
]

if __name__ == '__main__':

    for i in tqdm(params):
        Train(method=i.get('method'),
              batch_size=i.get('batch_size'))
