import numpy as np
import os

class DataReader:
    data_path = None
    label_path = None
    batch_size = None
    start_index = 0
    dummy = False
    _iteration = 0
    _epoch = 0

    def __init__(self, data_path, FLAGS, dtype):
        self.batch_size = FLAGS.batch_size
        self.data_path = data_path
        if dtype == 'train' or dtype == 'test':
            self.data_path = os.path.join(data_path, dtype + '_data.npy')
            self.label_path = os.path.join(data_path, dtype + '_label.npy')
        else:
            NameError('dtype not train or test')
        self._load_data()

    def _load_data(self):
        if self.dummy:
            N, H, W, C = (1000, 60, 150, 1)
            self.data = np.zeros([N, H, W, C], dtype=np.uint8)
            self.label = np.zeros(N, dtype=np.int32)
        else:
            self.data = np.load(self.data_path)
            self.label = np.load(self.label_path)
        if self.data.shape[0] < self.batch_size:
            raise NameError('batch size too large!')

    def next_batch(self):
        s = self.start_index
        e = min(self.data.shape[0], s + self.batch_size)

        data_batch = self.data[s:e]
        label_batch = self.label[s:e]

        self._iteration += 1

        if e == self.data.shape[0]:
            self.start_index = 0
            self._epoch += 1
        else:
            self.start_index = e
        return data_batch, label_batch

    def reset(self):
        self._iteration = 0
        self._epoch = 0

    def random_shuffle(self):
        N = self.data.shape[0]
        idx = np.random.permutation(N)
        self.data = self.data[idx]

    def shape(self):
        return self.data.shape

    @property
    def epoch(self):
        return self._epoch

    @property
    def iteration(self):
        return self._iteration
