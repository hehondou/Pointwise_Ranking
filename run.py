
from development.Pointwise.dataset import Dataset, get_trimmed_glove_vectors
from development.Pointwise.constants import *
from development.Pointwise.pointwise import Pointwise

def train():
    train_data = Dataset(ETYPE, DATASET)
    data_path = INPUT_PATH + 'train_dev_ctd_10.pickle'
    train_data.load_data(data_path)
    embeddings = get_trimmed_glove_vectors(INPUT_PATH + 'embedding_data.npz')

    model = Pointwise(embeddings, train_data)
    model.load_data()
    model.build()
    model.train()

def evaluate():
    model_params = {}
    model = Pointwise()
    model.build()

if __name__ == '__main__':
    train()


