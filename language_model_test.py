import random
import numpy as np
import tensorflow as tf
from language_model import LM
from hparams import HParams


def get_test_hparams():
    return HParams(
        batch_size=21,
        num_steps=12,
        num_shards=2,
        num_layers=1,
        learning_rate=0.2,
        max_grad_norm=1.0,

        vocab_size=1000,
        emb_size=14,
        state_size=17,
        projected_size=15,
        num_sampled=500,
        num_gpus=1,
        average_params=True,
        run_profiler=False,
    )


def simple_data_generator(batch_size, num_steps):
    x = np.zeros([batch_size, num_steps], np.int32)
    y = np.zeros([batch_size, num_steps], np.int32)
    for i in range(batch_size):
        first = random.randrange(0, 20)
        for j in range(num_steps):
            x[i, j] = first + j
            y[i, j] = first + j + 1
    return x, y, np.ones([batch_size, num_steps], np.uint8)


class TestLM(tf.test.test_util.TensorFlowTestCase):
    def test_lm(self):
        hps = get_test_hparams()

        with tf.variable_scope("model"):
            model = LM(hps)

        with self.test_session() as sess:
            tf.initialize_all_variables().run()
            tf.initialize_local_variables().run()

            loss = 1e5
            for i in range(50):
                x, y, w = simple_data_generator(hps.batch_size, hps.num_steps)
                loss, _ = sess.run([model.loss, model.train_op], {model.x: x, model.y: y, model.w: w})
                print("%d: %.3f %.3f" % (i, loss, np.exp(loss)))
                if np.isnan(loss):
                    print("NaN detected")
                    break

            self.assertLess(loss, 1.0)
