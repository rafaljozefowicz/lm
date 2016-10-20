import argparse
import os

parser = argparse.ArgumentParser(description="Run commands")
parser.add_argument("--logdir", type=str)
parser.add_argument("--hpconfig", type=str, default=",")
parser.add_argument("--datadir", type=str)


def new_tmux_cmd(name, cmd):
    if isinstance(cmd, (list, tuple)):
        cmd = " ".join(str(v) for v in cmd)
    return name, "tmux send-keys -t {} '{}' Enter".format(name, cmd)


def create_tmux_commands(session, gpus, logdir, hpconfig, datadir):
    cmds_map = []

    num_gpus = len(gpus)
    base_cmd = "python single_lm_train.py --logdir {}".format(logdir)
    gpus_str = ",".join(str(g) for g in gpus)

    cmds_map += [new_tmux_cmd(
        "worker", "CUDA_VISIBLE_DEVICES={} {} --num_gpus {} --hpconfig {} --datadir {}".format(
            gpus_str, base_cmd, num_gpus, hpconfig, datadir))]
    cmds_map += [new_tmux_cmd(
        "eval_testave", "CUDA_VISIBLE_DEVICES= {} --mode eval_test_ave --hpconfig {} --datadir {}".format(
            base_cmd, hpconfig, datadir))]
    cmds_map += [new_tmux_cmd("tb", ["tensorboard --logdir {} --port 12012".format(logdir)])]
    cmds_map += [new_tmux_cmd("htop", ["htop"])]

    windows = [v[0] for v in cmds_map]

    cmds = [
        "mkdir -p {}".format(logdir),
        "cd code/tf_dist",
        "tmux kill-session",
        "tmux new-session -s {} -n {} -d".format(session, windows[0])
    ]
    for w in windows[1:]:
        cmds += ["tmux new-window -t {} -n {}".format(session, w)]
    for window, cmd in cmds_map:
        cmds += [cmd]

    return cmds


def run():
    args = parser.parse_args()

    cmds = create_tmux_commands("lm1b", gpus=range(8), logdir=args.logdir, hpconfig=args.hpconfig, datadir=args.datadir)
    print("\n".join(cmds))
    os.system("\n".join(cmds))


if __name__ == "__main__":
    run()
