from src.train import *

params = [{'method': 'inception_v3', 'batch_size': 32, 'train_prop': 0.8}]

if __name__ == '__main__':
    Train(method=params[0].get('method'),
          batch_size=params[0].get('batch_size'),
          train_prop=params[0].get('train_prop'))