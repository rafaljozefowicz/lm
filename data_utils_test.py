import unittest

from data_utils import Vocabulary, Dataset


class DataUtilsTestCase(unittest.TestCase):
    def test_vocabulary(self):
        vocab = Vocabulary.from_file("testdata/test_vocab.txt")
        self.assertEqual(vocab.num_tokens, 1000)
        self.assertEqual(vocab.s_id, 2)
        self.assertEqual(vocab.s, "<S>")
        self.assertEqual(vocab.unk_id, 38)
        self.assertEqual(vocab.unk, "<UNK>")

    def test_dataset(self):
        vocab = Vocabulary.from_file("testdata/test_vocab.txt")
        dataset = Dataset(vocab, "testdata/*")

        def generator():
            for i in range(1, 10):
                yield [0] + list(range(1, i + 1)) + [0]
        counts = [0] * 10
        for seq in generator():
            for v in seq:
                counts[v] += 1

        counts2 = [0] * 10
        for x, y, w in dataset._iterate(generator(), 2, 4):
            for v in x.ravel():
                counts2[v] += 1
        for i in range(1, 10):
            self.assertEqual(counts[i], counts2[i], "Mismatch at i=%d" % i)

if __name__ == '__main__':
    unittest.main()
