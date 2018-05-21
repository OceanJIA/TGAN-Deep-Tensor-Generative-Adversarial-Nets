import os
import gzip
import urllib
import numpy as np
import pickle

def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='iso-8859-1')
    fo.close()
    return dict['data'], dict['labels']

def cifar_generator(filenames, batch_size, data_dir):
    all_data = []
    all_labels = []
    for filename in filenames:
        data, labels = unpickle(data_dir + '/' + filename)
        for i in range(len(labels)):
            all_data.append(data[i])
            all_labels.append(labels[i])

    images = all_data
    labels = all_labels

    def get_epoch():
        rng_state = np.random.get_state()
        np.random.shuffle(images)
        np.random.set_state(rng_state)
        np.random.shuffle(labels)

        for i in range(int(len(images) / batch_size)):
            yield (np.copy(images[i*batch_size:(i+1)*batch_size]), 
                  labels[i*batch_size:(i+1)*batch_size])

    return get_epoch


def load(batch_size, data_dir):
    return (
        cifar_generator(['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5'], batch_size, data_dir), 
        cifar_generator(['test_batch'], batch_size, data_dir)
)
