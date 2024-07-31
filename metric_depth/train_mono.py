# MIT License

# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Shariq Farooq Bhat

from zoedepth.utils.misc import count_parameters, parallelize
from zoedepth.utils.config import get_config
from zoedepth.utils.arg_utils import parse_unknown
from zoedepth.trainers.builder import get_trainer
from zoedepth.models.builder import build_model
from zoedepth.data.data_mono import DepthDataLoader
import torch.utils.data.distributed
import torch.multiprocessing as mp
import torch
import numpy as np
from pprint import pprint
import argparse
import os

os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["WANDB_START_METHOD"] = "thread"

from zoedepth.utils.easydict import EasyDict as edict


def fix_random_seed(seed: int):
    import random

    import numpy
    import torch

    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def load_ckpt(config, model, checkpoint_dir="./checkpoints", ckpt_type="best"):
    import glob
    import os

    print("Hello")

    from zoedepth.models.model_io import load_wts

    if hasattr(config, "checkpoint"):
        checkpoint = config.checkpoint
    elif hasattr(config, "ckpt_pattern"):
        pattern = config.ckpt_pattern
        matches = glob.glob(os.path.join(
            checkpoint_dir, f"*{pattern}*{ckpt_type}*"))
        if not (len(matches) > 0):
            raise ValueError(f"No matches found for the pattern {pattern}")

        checkpoint = matches[0]

    else:
        return model
    model = load_wts(model, checkpoint)
    print("Loaded weights from {0}".format(checkpoint))
    return model


def main_worker(gpu, ngpus_per_node, config):
    try:
        seed = config.seed if 'seed' in config and config.seed else 43
        fix_random_seed(seed)

        config.gpu = gpu

        model = build_model(config)
        # print(model)
            
        model = load_ckpt(config, model)
        model = parallelize(config, model)

        total_params = f"{round(count_parameters(model)/1e6,2)}M"
        config.total_params = total_params
        print(f"Total parameters : {total_params}")

        train_loader = DepthDataLoader(config, "train").data
        test_loader = DepthDataLoader(config, "online_eval").data

        trainer = get_trainer(config)(
            config, model, train_loader, test_loader, device=config.gpu)

        trainer.train()
    finally:
        import wandb
        wandb.finish()


if __name__ == '__main__':
    mp.set_start_method('spawn')

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="synunet")
    parser.add_argument("-d", "--dataset", type=str, default='nyu')
    parser.add_argument("--trainer", type=str, default=None)

    args, unknown_args = parser.parse_known_args()
    overwrite_kwargs = parse_unknown(unknown_args)

    overwrite_kwargs["model"] = args.model
    if args.trainer is not None:
        overwrite_kwargs["trainer"] = args.trainer

    config = {
        'attractor_alpha': 1000,
        'attractor_gamma': 2,
        'attractor_kind': 'mean',
        'attractor_type': 'inv',
        'aug': True,
        'avoid_boundary': False,
        'batch_size': 16,
        'bin_centers_type': 'softplus',
        'bin_embedding_dim': 128,
        'bs': 16,
        'clip_grad': 0.1,
        'cycle_momentum': True,
        'data_path': './data\\nyu',
        'data_path_eval': './data\\nyu',
        'dataset': 'nyu',
        'degree': 1.0,
        'dist_backend': 'nccl',
        'dist_url': 'tcp://127.0.0.1:15004',
        'distributed': True,
        'div_factor': 1,
        'do_kb_crop': False,
        'do_random_rotate': True,
        'eigen_crop': True,
        'encoder_lr_factor': 50,
        'epochs': 5,
        'filenames_file': './train_test_inputs/nyudepthv2_train_files_with_gt.txt',
        'filenames_file_eval': './train_test_inputs/nyudepthv2_test_files_with_gt.txt',
        'final_div_factor': 10000,
        'freeze_midas_bn': True,
        'garg_crop': False,
        'gpu': None,
        'gt_path': './data\\nyu',
        'gt_path_eval': './data\\nyu',
        'img_size': [392, 518],
        'input_height': 480,
        'input_width': 640,
        'inverse_midas': False,
        'log_images_every': 0.1,
        'lr': 0.000161,
        'max_depth': 10,
        'max_depth_diff': 10,
        'max_depth_eval': 10,
        'max_temp': 50.0,
        'max_translation': 100,
        'memory_efficient': True,
        'midas_lr_factor': 50,
        'midas_model_type': 'DPT_BEiT_L_384',
        'min_depth': 0.001,
        'min_depth_diff': -10,
        'min_depth_eval': 0.001,
        'min_temp': 0.0212,
        'mode': 'train',
        'model': 'zoedepth',
        'n_attractors': [16, 8, 4, 1],
        'n_bins': 64,
        'name': 'ZoeDepth',
        'ngpus_per_node': 0,
        'notes': '',
        'num_workers': 16,
        'output_distribution': 'logbinomial',
        'pct_start': 0.7,
        'pos_enc_lr_factor': 50,
        'prefetch': False,
        'pretrained_resource': '.\\checkpoints\\depth_anything_vitl14.pth',
        'print_losses': False,
        'project': 'MonoDepth3-nyu',
        'random_crop': False,
        'random_translate': False,
        'rank': 0,
        'root': '.',
        'same_lr': False,
        'save_dir': './depth_anything_finetune',
        'shared_dict': None,
        'tags': '',
        'three_phase': False,
        'train_midas': True,
        'trainer': 'zoedepth',
        'translate_prob': 0.2,
        'uid': None,
        'use_amp': False,
        'use_pretrained_midas': True,
        'use_shared_dict': False,
        'validate_every': 0.25,
        'version_name': 'v1',
        'w_domain': 0.2,
        'w_grad': 0,
        'w_reg': 0,
        'w_si': 1,
        'wd': 0.01,
        'workers': 16,
        'world_size': 1
        }
    config = edict(config)
    # config = get_config(args.model, "train", args.dataset, **overwrite_kwargs)
    # git_commit()
    if config.use_shared_dict:
        shared_dict = mp.Manager().dict()
    else:
        shared_dict = None
    config.shared_dict = shared_dict

    config.batch_size = config.bs
    config.mode = 'train'
    if config.root != "." and not os.path.isdir(config.root):
        os.makedirs(config.root)

    try:
        node_str = os.environ['SLURM_JOB_NODELIST'].replace(
            '[', '').replace(']', '')
        nodes = node_str.split(',')

        config.world_size = len(nodes)
        config.rank = int(os.environ['SLURM_PROCID'])
        # config.save_dir = "/ibex/scratch/bhatsf/videodepth/checkpoints"

    except KeyError as e:
        # We are NOT using SLURM
        config.world_size = 1
        config.rank = 0
        nodes = ["127.0.0.1"]

    if config.distributed:

        print(config.rank)
        port = np.random.randint(15000, 15025)
        config.dist_url = 'tcp://{}:{}'.format(nodes[0], port)
        print(config.dist_url)
        config.dist_backend = 'nccl'
        config.gpu = None

    ngpus_per_node = torch.cuda.device_count()
    config.num_workers = config.workers
    config.ngpus_per_node = ngpus_per_node
    print("Config:")
    pprint(config)
    if config.distributed:
        config.world_size = ngpus_per_node * config.world_size
        print(ngpus_per_node)
        mp.spawn(main_worker, nprocs=ngpus_per_node,
                 args=(ngpus_per_node, config))
    else:
        if ngpus_per_node == 1:
            config.gpu = 0
        main_worker(config.gpu, ngpus_per_node, config)
