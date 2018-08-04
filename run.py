
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
    test_path = 'development/Pointwise/data/disease/cdr/testdata_pw.pickle'
    test_data = Dataset(ETYPE, DATASET)
    test_data.load_data(test_path)
    embeddings = get_trimmed_glove_vectors(INPUT_PATH + 'embedding_data.npz')

    model = Pointwise(embeddings, test_data)
    model.build()
    model.load_data()
    model.transform()

if __name__ == '__main__':
    evaluate()


