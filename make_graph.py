import argparse
import os
import os.path as osp
import torch
import mesh2diff
import utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_args():
    """
    Args:
        config: json file with hyperparams and exp settings
        seed: random seed value
        stage: 1 for traing VAE, 2 for optimization,  and 12 for both
        logging: 
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default='params', help='config filename')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--logging', type=bool, default=True, help='logging')
    parser.add_argument('--stage', type=int, default=1, help='1.VAE, 2.BO, 12.VAE_BO, 3.Eval VAE')

    args = parser.parse_args()
    return args


args = parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# filename of the params
fname_config = args.config + '.json'
# read the params file
json_path = osp.join(osp.dirname(osp.realpath('__file__')), "config", fname_config)
hparams = utils.Params(json_path)
torch.cuda.set_device(hparams.device)

seq_len = hparams.seq_len
heart_torso = hparams.heart_torso

data_dir = osp.join(osp.dirname(osp.realpath('__file__')), 'data', 'training')

heart_names = hparams.heart_name
graph_names = hparams.graph_name
num_meshfrees = hparams.num_meshfree
structures = hparams.structures

for graph_name, heart_name, num_meshfree, structure in zip(graph_names, heart_names, num_meshfrees, structures):
    print(heart_name)
    root_dir = osp.join(data_dir, heart_name)
    graph_dir = osp.join(root_dir, 'raw', graph_name)
    # Create graph and load graph information
    g = mesh2diff.GraphPyramid(heart_name, structure, num_meshfree, seq_len)
    g.make_graph(heart_name)