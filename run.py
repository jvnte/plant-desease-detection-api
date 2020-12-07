from tqdm import tqdm
from src.train import *

params = [
    {'method': 'my_cnn', 'batch_size': 100, 'input_shape': (224, 224, 3)},
    {'method': 'vgg16', 'batch_size': 100, 'input_shape': (224, 224, 3)},
    {'method': 'mobile_net', 'batch_size': 100, 'input_shape': (224, 224, 3)}
]

if __name__ == '__main__':

    for i in tqdm(params):
        Train(method=i.get('method'),
              batch_size=i.get('batch_size'),
              input_shape=i.get('input_shape'))
