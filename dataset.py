import re
import pickle
import numpy as np
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from nltk.corpus import stopwords

from development.BioNen.constants import *
from utils import Timer

seed = 13
np.random.seed(seed)

STOPS = set(stopwords.words("english"))


class MyIOError(Exception):
    def __init__(self, filename):
        # custom error message
        message = """
        ERROR: Unable to locate file {}.

        FIX: Have you tried running python build_data.py first?
        This will build vocab file from your train, test and dev sets and
        trim your word vectors.""".format(filename)

        super(MyIOError, self).__init__(message)


def pre_process(text):
    """
    Remove punctuation, stop words; lowercase tokens
    """
    cleaned = re.sub("[^\w\d]", " ", text).lower()
    cleaned = re.sub(" +", " ", cleaned)

    tokens = cleaned.split()
    ret = [t for t in tokens if t not in STOPS]
    return " ".join(ret)


def tokenizer(text):
    return text.split()


def get_trimmed_glove_vectors(filename):
    """
    Args:
        filename: path to the npz file
    Returns:
        matrix of embeddings (np array)
    """
    try:
        with np.load(filename) as data:
            return data["embeddings"]

    except IOError:
        raise MyIOError(filename)


def load_vocab(filename):
    """
    Args:
        filename: file with a word per line
    Returns:
        d: dict[word] = index
    """
    try:
        d = dict()
        with open(filename, encoding='utf8') as f:
            for idx, word in enumerate(f):
                word = word.strip()
                d[word] = idx + 1  # preserve idx 0 for pad_tok

    except IOError:
        raise MyIOError(filename)
    return d


def pad_sequences(sequences, pad_tok, fix_length=None):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
        fix_length: minimum sequence's length
    Returns:
        a list of list where each sublist has same length
    """
    if fix_length:
        max_length = fix_length
    else:
        max_length = max(map(lambda x: len(x), sequences))
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
        sequence_padded += [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


class CandidateGenerator:
    def __init__(self):
        self.tf_idf = TfidfVectorizer(norm='l2',
                                      smooth_idf=False,
                                      tokenizer=tokenizer,
                                      analyzer='char_wb',
                                      ngram_range=(1, 3))

        self.id_to_names = None
        self.name_to_ids = None
        self.train_names = None
        self.tf_idf_names = None

    @staticmethod
    def _make_dicts(file_names):
        id_dict = {}
        rev_dict = {}

        raw_data = []
        for fn in file_names:
            with open(fn, 'r', encoding='utf8') as f:
                lines = f.read().strip().split('\n')
                raw_data.extend([l.split('\t') for l in lines])

        for d in raw_data:
            c = pre_process(d[1])
            # add term c to ID d[0]
            if not id_dict.get(d[0]):
                id_dict[d[0]] = set()
            id_dict[d[0]].add(c)
            # add ID d[0] to term c
            if not rev_dict.get(c):
                rev_dict[c] = set()
            rev_dict[c].add(d[0])

        return id_dict, rev_dict

    def train(self, train_file_names):
        t = Timer()
        t.start("Training candidate generator", verbal=True)
        self.id_to_names, self.name_to_ids = self._make_dicts(train_file_names)

        self.train_names = list(self.name_to_ids.keys())

        self.tf_idf_names = self.tf_idf.fit_transform(self.train_names)
        t.stop()

    def get_train_candidates(self, mention, id_, k):
        """
        :string mention: a name in the training vocabulary
        :string id_: the normalized ID
        :int k: even number of names in positive/negative list
        :return: 2-tuple of lists of positive and negative names
        """
        tf_idf_mention = self.tf_idf.transform([mention])

        # find positive candidates
        true_names = list(self.id_to_names[id_])
        tf_idf_true_names = self.tf_idf.transform(true_names)
        cosine_similarities = linear_kernel(tf_idf_mention, tf_idf_true_names)[0]

        sorted_indices = cosine_similarities.argsort()
        if cosine_similarities.size <= 2:  # use all
            true_name_indices = sorted_indices
        else:  # use k/2 closets and k/2 from the median
            sorted_indices = sorted_indices[:-1]  # remove the mention itself
            med = (len(sorted_indices) - 1) // 2
            med_upper = min(len(sorted_indices) - k // 2, med + k // 2)
            true_name_indices = list(sorted_indices[-k // 2:])[::-1] + list(sorted_indices[med:med_upper])[::-1]
        pos_candidates = []
        for t in true_name_indices:
            pos_candidates.append(true_names[t])

        # find negative candidates
        neg_candidates = []
        cosine_similarities = linear_kernel(tf_idf_mention, self.tf_idf_names)[0]

        name_indices = cosine_similarities.argsort()[::-1]
        j = 0
        k = 0
        while j < len(pos_candidates) and k < len(name_indices):
            n = self.train_names[name_indices[k]]
            n_ids = self.name_to_ids[n]
            if id_ not in n_ids:
                neg_candidates.append(n)
                j += 1
            k += 1

        return pos_candidates, neg_candidates


class Dataset:
    def __init__(self, e_type, dataset, use_dev=False, k=4):
        self.e_type = e_type
        self.dataset = dataset
        self.data = {}
        self._load_vocabs()
        self.max_term_length = 0
        self.max_word_length = 0
        self.use_dev = use_dev
        self.k = k
        self.candidate_generator = None
        self.train_names = None
        self.name_to_ids = None

    def create(self):
        t = Timer()
        t.start("Creating dataset", verbal=True)

        self.candidate_generator = CandidateGenerator()
        ph = DATA + self.e_type + "/" + self.dataset + "/{}" + "_id_term.txt"
        if self.use_dev:
            self.candidate_generator.train(["{}{}/ctd_id_term.txt".format(DATA, self.e_type),
                                            ph.format("train"), ph.format("dev")])
        else:
            self.candidate_generator.train(["{}{}/ctd_id_term.txt".format(DATA, self.e_type),
                                            ph.format("train")])

        idx = 0
        for m in self.candidate_generator.train_names:
            for i in self.candidate_generator.name_to_ids[m]:
                pos_n, neg_n = self.candidate_generator.get_train_candidates(m, i, self.k)
                self.data[idx] = {
                    'x_m': self.process_word(m),
                    'x_n': [],
                    'y': [],
                    'id': i,
                    'm': m
                }
                for n in pos_n:
                    x_n = self.process_word(n)
                    self.data[idx]['x_n'].append(x_n)
                    self.data[idx]['y'].append(1)
                for n in neg_n:
                    x_n = self.process_word(n)
                    self.data[idx]['x_n'].append(x_n)
                    self.data[idx]['y'].append(0)
            idx += 1
        t.stop()

    def save(self, file_name):
        t = Timer()
        t.start("Saving dataset", verbal=True)
        print("Max term length: {}\nMax word length: {}".format(self.max_term_length, self.max_word_length))
        with open(file_name, 'wb') as f:
            pickle.dump((self.max_term_length, self.max_word_length), f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.data, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.candidate_generator.train_names, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.candidate_generator.name_to_ids, f, pickle.HIGHEST_PROTOCOL)
        t.stop()

    def load_data(self, file_name):
        with open(file_name, 'rb') as f:
            self.max_term_length, self.max_word_length = pickle.load(f)
            self.data = pickle.load(f)
            self.train_names = pickle.load(f)
            self.name_to_ids = pickle.load(f)

    def get_num_chars(self):
        path = "{}{}/{}/all_chars.txt".format(DATA, self.e_type, self.dataset)
        with open(path, 'r', encoding='utf8') as f:
            d = f.readlines()
        return len(d)

    def _load_vocabs(self):
        self.vocab_words = load_vocab("{}{}/{}/all_words.txt".format(DATA, self.e_type, self.dataset))
        self.vocab_chars = load_vocab("{}{}/{}/all_chars.txt".format(DATA, self.e_type, self.dataset))

    def process_word(self, term):
        words = tokenizer(term)

        char_ids = []
        # 1. get chars of words
        for char in term:
            # ignore chars out of vocabulary
            if char in self.vocab_chars:
                char_ids.append(self.vocab_chars[char])

        # 2. get id of word
        word_ids = []
        for w in words:
            if w in self.vocab_words:
                word_ids.append(self.vocab_words[w])
            else:
                word_ids.append(self.vocab_words[UNK])

        # 3. find max word and term length
        if len(char_ids) > self.max_word_length:
            self.max_word_length = len(char_ids)
        if len(word_ids) > self.max_term_length:
            self.max_term_length = len(word_ids)

        # 4. return tuple char ids, word id
        return char_ids, word_ids


class GoldenTestDataset(Dataset):
    def __init__(self, e_type, dataset, filename):
        super().__init__(e_type, dataset)
        self.filename = filename
        self.raw_data = []
        self.X = []
        self.true_ids = {}

    def create(self):
        with open(self.filename, 'r') as f:
            lines = f.readlines()

        regex = re.compile(r'^(\d+)\t(\d+)\t(\d+)\t([^\t]+)\t(\S+)\t(\S+)', re.U | re.I)

        # create raw data
        for l in lines:
            matched = regex.match(l)
            if matched:
                data = matched.groups()
                pid = data[0]
                offset = (data[1], data[2])
                c = data[3]
                t = data[4]
                i = data[5]
                if i == '-1':  # no mesh id for this concept
                    continue

                # using only the first ID as label
                if (self.e_type == "chemical" and t == 'Chemical') or (self.e_type == "disease" and t != 'Chemical'):
                    self.raw_data.append((pid, offset, pre_process(c), set(i.split('|'))))

        # create X and true_ids and dummy pred_ids
        for m in self.raw_data:
            self.X.append(self.process_word(m[2]))
            if not self.true_ids.get(m[0]):
                self.true_ids[m[0]] = []
            self.true_ids[m[0]].append([m[1], m[3]])


class NERTestDataset(Dataset):
    def __init__(self, e_type, dataset, golden_test, ner_output):
        super().__init__(e_type, dataset)
        self.ner_output = ner_output
        self.golden_test = golden_test
        self.raw_data = []
        self.golden_data = []
        self.X = []
        self.true_ids = {}

    def create(self):
        with open(self.ner_output, 'r') as f:
            ner_lines = f.readlines()
        with open(self.golden_test, 'r') as f:
            golden_lines = f.readlines()

        ner_regex = re.compile(r'^(\d+)\t(\d+)\t(\d+)\t([^\t]+)\t(\S+)', re.U | re.I)
        golden_regex = re.compile(r'^(\d+)\t(\d+)\t(\d+)\t([^\t]+)\t(\S+)\t(\S+)', re.U | re.I)

        # create raw data with ner lines
        for l in ner_lines:
            matched = ner_regex.match(l)
            if matched:
                data = matched.groups()
                pid = data[0]
                offset = (data[1], data[2])
                c = data[3]
                t = data[4]
                if (self.e_type == "chemical" and t == 'Chemical') or (self.e_type == "disease" and t != 'Chemical'):
                    self.raw_data.append((pid, offset, pre_process(c)))

        # create golden data with golden lines
        for l in golden_lines:
            matched = golden_regex.match(l)
            if matched:
                data = matched.groups()
                pid = data[0]
                offset = (data[1], data[2])
                t = data[4]
                i = data[5]
                if i == '-1':  # no mesh id for this concept
                    continue

                # using only the first ID as label
                if (self.e_type == "chemical" and t == 'Chemical') or (self.e_type == "disease" and t != 'Chemical'):
                    self.golden_data.append((pid, offset, set(i.split('|'))))

        # create X
        for m in self.raw_data:
            self.X.append(self.process_word(m[2]))
        # create true_ids
        for m in self.golden_data:
            if not self.true_ids.get(m[0]):
                self.true_ids[m[0]] = []
            self.true_ids[m[0]].append([m[1], m[2]])


def main(entity_type, dataset, outfile, use_dev, k):
    d = Dataset(entity_type, dataset, use_dev=use_dev, k=k)
    d.create()
    d.save("{}{}/{}/{}.pickle".format(DATA, entity_type, dataset, outfile))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build necessary data for model training and evaluating.')
    parser.add_argument('entity_type', help="type of the entity, i.e: disease")
    parser.add_argument('dataset', help="the name of the dataset that the model will be trained on, i.e: cdr")
    parser.add_argument('output', help="name of the output data file, i.e: train_ctd")
    parser.add_argument('k', help="the number of names in each positive and negative list, i.e: 10", type=int)
    parser.add_argument("-dev", "--use_dev", help="use development set", action="store_true")

    args = parser.parse_args()

    main(args.entity_type, args.dataset, args.output, args.use_dev, args.k)
