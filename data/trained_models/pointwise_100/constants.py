# Model Hyperparameters
WORD_EMBEDDING_DIM = 200
FILTER_SIZES = [1, 2, 3, 5]
NUM_FILTERS = 128
DROPOUT_KEEP_PROB = 0.5
L2_REG_LAMBDA = 1e-6
LEARNING_RATE = 1e-3
MAX_INPUT_WORD = 15
HIDDEN_NUM = 20

# Training parameters
BATCH_SIZE = 64
TRAINABLE = False
NUM_EPOCHS = 20
IS_EARLY_STOPPING = False

# Folders' names
DATA = "development/Pointwise/data/"
TRAINED_MODELS = "development/Pointwise/data/trained_models/"

# Word embedding configuration
WORD_EMBEDDING = "data/w2v_model/wikipedia-pubmed-and-PMC-w2v.bin"

# Name parameters
ETYPE = 'disease'
DATASET = 'cdr'
MODEL_NAME = 'pointwise_100'

INPUT_PATH = DATA + '{}/{}/'.format(ETYPE, DATASET)
MODEL_PATH = TRAINED_MODELS + MODEL_NAME + "/"