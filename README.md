# lm

The codebase implements LSTM language model baseline from https://arxiv.org/abs/1602.02410
The code supports running on the machine with multiple GPUs using synchronized gradient updates (which is the main difference with the paper).

The code was tested on a box with 8 Geforce Titan X and LSTM-2048-512 (default configuration) can process up to 100k words per second.
The perplexity on the holdout set after 5 epochs is about 48.7 (vs 47.5 in the paper), which can be due to slightly different hyper-parameters.
It takes about 16 hours to reach these results on 8 Titan Xs. DGX-1 is about 30% faster on the baseline model.


## Dependencies
* Anaconda
* TensorFlow 0.10
* Python 3.5 (should work with 2.7 but haven't tested it recently)
* 1B Word Benchmark Dataset (https://github.com/ciprian-chelba/1-billion-word-language-modeling-benchmark to get data)
* tmux (the start script opens up a tmux session with multiple windows)


## To run
Assuming the data directory is in: `/home/rafal/datasets/lm1b/`, execute:

`python single_lm_run.py --datadir /home/rafal/datasets/lm1b/ --logdir <log_dir>`

It'll start a tmux session and you can connect to it with: `tmux a`. It should contain several windows:
* (window:0) training worker
* (window:1) evaluation script
* (window:2) tensorboard
* (window:3) htop

The scripts above executes the following commands, which can be run manually:
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python single_lm_train.py --logdir <log_dir> --num_gpus 8 --datadir <data_dir>
CUDA_VISIBLE_DEVICES= python single_lm_train.py --logdir <log_dir> --mode eval_test_ave --datadir <data_dir>
tensorboard --logdir <log_dir> --port 12012
```

Please note that this assumes the user has 8 GPUs available. Changing the CUDA_VISIBLE_DEVICES mask and --num_gpus flag to something else will work but the training will obviously be slower.


Results can be monitored using TensorBoard, listening on port 12012.

## To change hyper-parameters

The command accepts and additional argument `--hpconfig` which allows to override various hyper-parameters, including:
* batch_size=128 - batch size
* num_steps=20 - number of unrolled LSTM steps
* num_shards=8 -  embedding and softmax matrices are split into this many shards
* num_layers=1 - number of LSTM layers
* learning_rate=0.2 - learning rate for adagrad
* max_grad_norm=10.0 - maximum acceptable gradient norm 
* keep_prob=0.9 - for dropout between layers (here: 10% dropout before and after each LSTM layer)
* emb_size=512 - size of the embedding
* state_size=2048 - LSTM state size
* projected_size=512 - LSTM projection size 
* num_sampled=8192 - number of word target samples for IS objective during training

To run a version of the model with 2 layers and 4096 state size, simply call:

`python single_lm_run.py --datadir /home/rafal/datasets/lm1b/ --logdir <log_dir> --hpconfig num_layers=2,state_size=4096`


## Feedback
Let me know if you have any questions or comments at rafjoz@gmail.com
