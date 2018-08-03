from development.Pointwise.constants import *
from development.Pointwise.dataset import pad_sequences

from utils import Timer
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import numpy as np
import tensorflow as tf
import os
import shutil
shutil.copy2('development/Pointwise/constants.py', MODEL_PATH)

seed = 13
np.random.seed(seed)

class Pointwise:
    def __init__(self, embeddings, data):
        self.embeddings = embeddings
        self.data = data

        self.word_embedding_dim = WORD_EMBEDDING_DIM
        self.filter_sizes = FILTER_SIZES
        self.num_filters = NUM_FILTERS
        self.dropout_keep_prob = DROPOUT_KEEP_PROB
        self.l2_reg_lambda = L2_REG_LAMBDA
        self.learning_rate = LEARNING_RATE
        self.max_input_word = MAX_INPUT_WORD
        self.hidden_num = HIDDEN_NUM

        self.batch_size = BATCH_SIZE
        self.trainable = TRAINABLE
        self.num_epochs = NUM_EPOCHS
        self.is_early_stopping = IS_EARLY_STOPPING

        self.para = []

    def load_data(self):
        timer = Timer()
        timer.start("Loading start")

        self.data = self.data.data
        all_idx = list(self.data.keys())

        if self.is_early_stopping:
            self.train_idx, self.dev_idx = train_test_split(all_idx, test_size=0.3, random_state=seed)
            print("Number of validating mentions: {}".format(len(self.dev_idx)))
        else:
            self.train_idx = all_idx
        print("Number of training mentions: {}".format(len(self.train_idx)))

        timer.stop()

    def build(self):
        self._add_placeholder()
        self._add_model_op()
        self._add_train_op()

    def _add_placeholder(self):
        self.m_word_ids = tf.placeholder(tf.int32, [None, self.max_input_word], name="m_word_ids")
        self.n_word_ids = tf.placeholder(tf.int32, [None, self.max_input_word], name="n_word_ids")

        self.y = tf.placeholder(tf.int32, [None, 2], name="input_y")
        self.idx = tf.placeholder(tf.int32, [None], name="idx")

        self.dropout_keep_op = tf.placeholder(tf.float32, name="dropout_keep_prob")

    def _add_model_op(self):
        # embeddings
        with tf.name_scope('embeddings'):
            _word_embeddings = tf.Variable(self.embeddings, name="word_embeddings", dtype=tf.float32, trainable=TRAINABLE)
            self.m_word_embedding = self._get_embedding(_word_embeddings, self.m_word_ids)
            self.n_word_embedding = self._get_embedding(_word_embeddings, self.n_word_ids)

        # convolution
        self.kernels = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope('conv-pool-%s' % filter_size):
                filter_shape = [filter_size, self.word_embedding_dim, 1, self.num_filters]
                W = tf.get_variable('W' + str(i), filter_shape, tf.float32,
                                    tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN', uniform=True))
                b = tf.get_variable('b' + str(i), [self.num_filters], tf.float32, tf.constant_initializer(0.01))
                self.kernels.append((W, b))
                self.para.append(W)
                self.para.append(b)

        self.num_filters_total = self.num_filters * len(self.filter_sizes)

        self.m_conv = self._convolution(self.m_word_embedding)
        self.n_conv = self._convolution(self.n_word_embedding)

        # max pooling
        with tf.name_scope('max_pooling'):
            self.m_pooling = tf.reshape(self._max_pooling(self.m_conv, self.max_input_word), [-1, self.num_filters_total])
            self.n_pooling = tf.reshape(self._max_pooling(self.n_conv, self.max_input_word), [-1, self.num_filters_total])

        with tf.variable_scope("similarity"):
            W = tf.get_variable(
                name="W",
                shape=[self.num_filters_total, self.num_filters_total],
                initializer=tf.contrib.layers.xavier_initializer()
            )
            self.transform_left = tf.matmul(self.m_pooling, W)
            self.sims = tf.reduce_sum(tf.multiply(self.transform_left, self.n_pooling), axis=1, keepdims=True)
            self.para.append(W)
            self.see = W

        self.feature = tf.concat([self.m_pooling, self.n_pooling], axis=1, name="feature")

        with tf.name_scope('neural_network'):
            W = tf.get_variable(
                "W_hidden",
                shape=[2 * self.num_filters_total, self.hidden_num],
                initializer=tf.contrib.layers.xavier_initializer()
            )
            b = tf.get_variable('b_hidden', shape=[self.hidden_num], initializer=tf.random_normal_initializer())
            self.para.append(W)
            self.para.append(b)
            self.hidden_output = tf.nn.relu(tf.nn.xw_plus_b(self.feature, W, b, name="hidden_output"))

        with tf.name_scope('dropout'):
            self.h_drop = tf.nn.dropout(self.hidden_output, self.dropout_keep_prob, name="hidden_ouput_drop")

        with tf.name_scope('output'):
            W = tf.get_variable(
                "W_output",
                shape=[self.hidden_num, 2],
                initializer=tf.contrib.layers.xavier_initializer()
            )
            b = tf.get_variable('b_output', shape=[2], initializer=tf.random_normal_initializer())
            self.para.append(W)
            self.para.append(b)
            self.logits = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.scores = tf.nn.softmax(self.logits)
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

    def _add_train_op(self):
        l2_loss = tf.constant(0.0)
        for p in self.para:
            l2_loss += tf.nn.l2_loss(p)
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.y)
            self.loss_op = tf.reduce_mean(losses) + self.l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        global_step = tf.Variable(0, name="global_step", trainable=False)
        starter_learning_rate = self.learning_rate
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 100, 0.96)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        grads_and_vars = optimizer.compute_gradients(self.loss_op)
        self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    def train(self):
        self._train(self.num_epochs)

    def _train(self, epochs, patience=4):

        if not os.path.exists(MODEL_PATH):
            os.makedirs(MODEL_PATH)
        saver = tf.train.Saver(max_to_keep=1)

        timer = Timer()
        timer.start('Start training')

        best_acc = 0
        nepoch_noimp = 0
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            num_batch_train = len(self.train_idx) // self.batch_size + 1
            for e in range(epochs):
                total_train_loss = 0
                total_train_acc = 0
                c = 0
                train_idx = shuffle(self.train_idx)
                for idx, batch in enumerate(self._next_batch(data=train_idx, num_batch=num_batch_train)):
                    feed_dict = self._make_feed_dict(batch, self.dropout_keep_prob)
                    _, train_loss, predictions, input_y, train_acc = sess.run([self.train_op, self.loss_op,
                                                                               self.predictions, self.y,
                                                                               self.accuracy],
                                                                              feed_dict=feed_dict)
                    total_train_loss += train_loss
                    total_train_acc += train_acc
                    c += 1

                    if idx % 200 == 0:
                        print("Iter {} - Loss: {} - Acc: {}".format(idx, total_train_loss / c, total_train_acc / c))
                print("End epochs {} ".format(e + 1))

                if self.is_early_stopping:
                    num_batch_val = len(self.dev_idx) // self.batch_size + 1
                    total_val_loss = 0
                    total_val_acc = 0
                    c = 0
                    for idx, batch in enumerate(self._next_batch(data=self.dev_idx, num_batch=num_batch_val)):
                        feed_dict = self._make_feed_dict(batch, self.dropout_keep_prob)
                        _, dev_loss, dev_acc = sess.run([self.train_op, self.loss_op, self.accuracy], feed_dict=feed_dict)
                        total_val_loss += dev_loss
                        total_val_acc += dev_acc
                        c += 1
                    avg_loss = total_val_loss / c
                    avg_acc = total_val_acc / c
                    print("Validation Loss: {} - Acc: {}".format(avg_loss, avg_acc))

                    if avg_acc > best_acc:
                        saver.save(sess, MODEL_PATH)
                        print('Saved the model at epoch {}'.format(e + 1))
                        best_acc = avg_acc
                        nepoch_noimp = 0
                    else:
                        nepoch_noimp += 1
                        print("Number of epochs with no improvement: {}".format(nepoch_noimp))
                        if nepoch_noimp >= patience:
                            break
            if not self.is_early_stopping:
                saver.save(sess, MODEL_PATH)
                print("Saved the model")

        timer.stop()

    def _next_batch(self, data, num_batch):
        start = 0
        c = 0

        while c < num_batch:
            index_batch = data[start:start + self.batch_size]

            X_m_batch = []
            X_n_batch = []
            Y_batch = []
            IDX_batch = []

            for i in index_batch:
                num = len(self.data[i]['y'])
                y_data = self.data[i]['y']
                for y in y_data:
                    if y == 1:
                        Y_batch.append([0, 1])
                    else:
                        Y_batch.append([1, 0])
                IDX_batch.extend([i] * num)
                X_n_batch.extend(self.data[i]['x_n'])
                X_m_batch.extend([self.data[i]['x_m']] * num)

            X_m_batch, X_n_batch, Y_batch, IDX_batch = shuffle(X_m_batch, X_n_batch, Y_batch, IDX_batch)

            m_word_ids = self._get_word_ids(X_m_batch)
            n_word_ids = self._get_word_ids(X_n_batch)

            start += self.batch_size
            c +=1
            yield(m_word_ids, n_word_ids, Y_batch, IDX_batch)

    def _make_feed_dict(self, data, dropout_keep_prob):
        return {
            self.m_word_ids: data[0],
            self.n_word_ids: data[1],
            self.y: data[2],
            self.idx: data[3],
            self.dropout_keep_op: dropout_keep_prob
        }

    def _get_word_ids(self, X):
        word_ids = [x[1] for x in X]
        word_ids, _ = pad_sequences(word_ids, pad_tok=0, fix_length=self.max_input_word)
        return word_ids

    def _get_embedding(self, embeddings, ids):
        embed = tf.nn.embedding_lookup(embeddings, ids)
        embed = tf.reshape(embed, [-1, self.max_input_word, self.word_embedding_dim, 1])
        return embed

    def _convolution(self, embedding):
        cnn_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            conv = tf.nn.conv2d(
                embedding,
                self.kernels[i][0],
                strides=[1, 1, self.word_embedding_dim, 1],
                padding='SAME',
                name='conv-1'
            )
            h = tf.nn.relu(tf.nn.bias_add(conv, self.kernels[i][1]), name='relu-1')
            cnn_outputs.append(h)
        cnn_reshaped = tf.concat(cnn_outputs, 3)
        return cnn_reshaped

    def _max_pooling(self, conv, input_length):
        pooled = tf.nn.max_pool(
            conv,
            ksize=[1, input_length, 1, 1],
            strides=[1, 1, 1, 1],
            padding='VALID',
            name="pool")
        return pooled




