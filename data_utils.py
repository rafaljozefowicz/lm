import codecs
import glob
import json
import random

import numpy as np


class Vocabulary(object):

    def __init__(self):
        self._token_to_id = {}
        self._token_to_count = {}
        self._id_to_token = []
        self._num_tokens = 0
        self._s_id = None
        self._unk_id = None

    @property
    def num_tokens(self):
        return self._num_tokens

    @property
    def unk(self):
        return "<UNK>"

    @property
    def unk_id(self):
        return self._unk_id

    @property
    def s(self):
        return "<S>"

    @property
    def s_id(self):
        return self._s_id

    def add(self, token, count):
        self._token_to_id[token] = self._num_tokens
        self._token_to_count[token] = count
        self._id_to_token.append(token)
        self._num_tokens += 1

    def finalize(self):
        self._s_id = self.get_id(self.s)
        self._unk_id = self.get_id(self.unk)

    def get_id(self, token):
        return self._token_to_id.get(token, self.unk_id)

    def get_token(self, id_):
        return self._id_to_token[id_]

    @staticmethod
    def from_file(filename):
        vocab = Vocabulary()
        with codecs.open(filename, "r", "utf-8") as f:
            for line in f:
                word, count = line.strip().split()
                vocab.add(word, int(count))
        vocab.finalize()
        return vocab


class Dataset(object):

    def __init__(self, vocab, file_pattern, deterministic=False):
        self._vocab = vocab
        self._file_pattern = file_pattern
        self._deterministic = deterministic

    def _parse_sentence(self, line):
        s_id = self._vocab.s_id
        return [s_id] + [self._vocab.get_id(word) for word in line.strip().split()] + [s_id]

    def _parse_file(self, file_name):
        print("Processing file: %s" % file_name)
        with codecs.open(file_name, "r", "utf-8") as f:
            lines = [line.strip() for line in f]
            if not self._deterministic:
                random.shuffle(lines)
            print("Finished processing!")
            for line in lines:
                yield self._parse_sentence(line)

    def _sentence_stream(self, file_stream):
        for file_name in file_stream:
            for sentence in self._parse_file(file_name):
                yield sentence

    def _iterate(self, sentences, batch_size, num_steps):
        streams = [None] * batch_size
        x = np.zeros([batch_size, num_steps], np.int32)
        y = np.zeros([batch_size, num_steps], np.int32)
        w = np.zeros([batch_size, num_steps], np.uint8)
        while True:
            x[:] = 0
            y[:] = 0
            w[:] = 0
            for i in range(batch_size):
                tokens_filled = 0
                try:
                    while tokens_filled < num_steps:
                        if streams[i] is None or len(streams[i]) <= 1:
                            streams[i] = next(sentences)
                        num_tokens = min(len(streams[i]) - 1, num_steps - tokens_filled)
                        x[i, tokens_filled:tokens_filled+num_tokens] = streams[i][:num_tokens]
                        y[i, tokens_filled:tokens_filled + num_tokens] = streams[i][1:num_tokens+1]
                        w[i, tokens_filled:tokens_filled + num_tokens] = 1
                        streams[i] = streams[i][num_tokens:]
                        tokens_filled += num_tokens
                except StopIteration:
                    pass
            if not np.any(w):
                return

            yield x, y, w

    def iterate_once(self, batch_size, num_steps):
        def file_stream():
            for file_name in glob.glob(self._file_pattern):
                yield file_name
        for value in self._iterate(self._sentence_stream(file_stream()), batch_size, num_steps):
            yield value

    def iterate_forever(self, batch_size, num_steps):
        def file_stream():
            while True:
                file_patterns = glob.glob(self._file_pattern)
                if not self._deterministic:
                    random.shuffle(file_patterns)
                for file_name in file_patterns:
                    yield file_name
        for value in self._iterate(self._sentence_stream(file_stream()), batch_size, num_steps):
            yield value
