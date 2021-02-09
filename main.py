import argparse
import os
import os.path as osp
from shutil import copy2

import torch
from torch import optim
import numpy as np

import scipy.io
# import mesh2
import net
import train
import utils
from torch_geometric.data import DataLoader
from data_utils import HeartGraphDataset

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

    parser.add_argument('--config', type=str, default='p16', help='config filename')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--logging', type=bool, default=True, help='logging')
    parser.add_argument('--stage', type=int, default=1, help='1.VAE, 2.BO, 12.VAE_BO, 3.Eval VAE')
    parser.add_argument('--checkpt', type=str, default='None', help='checkpoint to resume training from')

    args = parser.parse_args()
    return args


def learn_vae_heart_torso(hparams, checkpt, training=True, fine_tune=False):
    """Generative modeling of the HD tissue properties
    """
    vae_type = hparams.model_type
    batch_size = hparams.batch_size if training else 1
    num_epochs = hparams.num_epochs
    seq_len = hparams.seq_len
    heart_torso = hparams.heart_torso
    anneal = hparams.anneal

    # directory path for training and testing datasets
    data_dir = osp.join(osp.dirname(osp.realpath('__file__')),
                        'data', 'training')
    phy_dir = osp.join(osp.dirname(osp.realpath('__file__')),
                        'data', 'phy_vars')

    # directory path to save the model/results
    model_dir = osp.join(osp.dirname(osp.realpath('__file__')),
                         'experiments', vae_type, hparams.model_name)
    if not osp.exists(model_dir):
        os.makedirs(model_dir)
    
    if training:
        copy2(net_path, model_dir)
    copy2(json_path, model_dir)

    # logging the training procedure
    # if args.logging:
    #    sys.stdout = open(model_dir+'/log.txt','wt')

    corMfrees = dict()
    train_loaders = dict()
    test_loaders = dict()
    heart_names = hparams.heart_name
    graph_method = hparams.graph_method
    num_meshfrees = hparams.num_meshfree
    structures = hparams.structures
    sample = hparams.sample if training else 1
    subset = hparams.subset if training else 1
    learning_rate = hparams.learning_rate
    epoch_start = 1
    # subset = 1

    # initialize the model
    if hparams.net_arch == 'phy':
        model = net.GraphTorsoHeart(hparams, training=training)
    elif hparams.net_arch == 'latent_ode':
        model = net.Graph_LODE(hparams, training=training)
    elif hparams.net_arch == 'ode_rnn':
        model = net.Graph_ODE_RNN(hparams, training=training)
    elif hparams.net_arch == 'ode_rnn_embedding':
        model = net.Graph_ODE_RNN_Embedding(hparams, training=training)
    else:
        raise NotImplementedError('The architecture {} is not implemented'.format(hparams.net_arch))
    
    for heart_name, num_meshfree, structure in zip(heart_names, num_meshfrees, structures):
        root_dir = osp.join(data_dir, heart_name)
        graph_dir = osp.join(root_dir, 'raw/{}_{}'.format(heart_name, graph_method))
        # import ipdb; ipdb.set_trace()
        graphparams = net.get_graphparams(graph_dir, device, batch_size, heart_torso, graph_method)

        # initialize datasets and dataloader
        train_dataset = HeartGraphDataset(root=root_dir, num_meshfree=num_meshfree, seq_len=seq_len,
                                        mesh_graph=graphparams["g"], mesh_graph_torso=graphparams["t_g"],
                                        heart_torso=heart_torso, train=True, subset=subset)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=training, drop_last=True)
        
        corMfrees[heart_name] = train_dataset.getCorMfree()
        train_loaders[heart_name] = train_loader

        model.set_graphs(graphparams, heart_name)
        
        h_L, t_L, H = net.get_physics(phy_dir, heart_name, device)
        model.set_physics(h_L, t_L, H, heart_name)

    if training:
        val_heart = hparams.val_heart
        val_meshfree = hparams.val_meshfree
        val_structures = hparams.val_structures
        root_dir = osp.join(data_dir, val_heart[0])

        graph_dir = osp.join(root_dir, 'raw/{}_{}'.format(val_heart[0], graph_method))
        graphparams = net.get_graphparams(graph_dir, device, batch_size, heart_torso, graph_method)
        # state = not fine_tune
        state = False
        test_dataset = HeartGraphDataset(root=root_dir, num_meshfree=val_meshfree[0], seq_len=seq_len,
                                        mesh_graph=graphparams["g"], mesh_graph_torso=graphparams["t_g"],
                                        heart_torso=heart_torso, train=state)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        test_loaders[val_heart[0]] = test_loader
        model.set_graphs(graphparams, val_heart[0])

        h_L, t_L, H = net.get_physics(phy_dir, val_heart[0], device)
        model.set_physics(h_L, t_L, H, val_heart[0])
        if fine_tune:
            pre_model_dir = osp.join(osp.dirname(osp.realpath('__file__')), 'experiments', vae_type, hparams.pre_model_name)
            model.load_state_dict(torch.load(pre_model_dir + '/' + hparams.vae_latest, map_location=device))
        
        model.to(device)
        # loss_function = net.loss_stgcnn
        loss_function = net.loss_stgcnn_mixed

        phy_mode = hparams.phy_mode
        smooth = hparams.smooth
        hidden = hparams.hidden

        # Set checkpoint values to model and learning rate
        if checkpt is not None:
            model.load_state_dict(checkpt['state_dict'])
            learning_rate = checkpt['cur_learning_rate']
            epoch_start = checkpt['epoch'] + 1

        # Set up optimizer and LR scheduler
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        if checkpt is not None:
            optimizer.load_state_dict(checkpt['optimizer'])

        if hparams.gamma < 1.0:
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=hparams.gamma, patience=10,
                                                                  min_lr=5e-5, verbose=True)
        else:
            print("Using StepLR with no decay.")
            lr_scheduler = None

        # Run the training loop
        train.train_vae(model, checkpt, epoch_start, optimizer, lr_scheduler, train_loaders, test_loaders, loss_function, phy_mode, smooth,
                        hidden, model_dir, num_epochs, batch_size, seq_len, corMfrees, anneal, sample)
    else:
        if checkpt is None:
            checkpt = torch.load(model_dir + '/' + hparams.vae_latest, map_location=device)
        model.load_state_dict(checkpt['state_dict'])
        model = model.eval().to(device)
        train.eval_vae(model, train_loaders, model_dir, batch_size, seq_len, corMfrees)
        # train_heart_torso.eval_real_new(model, train_loaders, exp_dir, corMfrees)


def real_data_new(hparams, training=False):
    vae_type = hparams.model_type
    batch_size = hparams.batch_size
    num_epochs = hparams.num_epochs
    seq_len = hparams.seq_len
    heart_torso = hparams.heart_torso
    anneal = hparams.anneal

    corMfrees = dict()
    train_loaders = dict()
    test_loaders = dict()
    heart_names = hparams.heart_name
    graph_method = hparams.graph_method
    num_meshfrees = hparams.num_meshfree
    structures = hparams.structures
    sample = hparams.sample if training else 1

    data_dir = osp.join(osp.dirname(osp.realpath('__file__')), 'data', 'training')
    model_dir = osp.join(osp.dirname(osp.realpath('__file__')), 'experiments', vae_type, hparams.pre_model_name)
    exp_dir = osp.join(osp.dirname(osp.realpath('__file__')), 'experiments', vae_type, hparams.model_name)
    if not osp.exists(exp_dir):
        os.makedirs(exp_dir)
    if training:
        copy2(net_path, exp_dir)
    copy2(json_path, exp_dir)

    model = net.GraphTorsoHeart(hparams)

    for heart_name, num_meshfree, structure in zip(heart_names, num_meshfrees, structures):
        root_dir = osp.join(data_dir, heart_name)
        graph_dir = osp.join(root_dir, 'raw/{}_{}'.format(heart_name, graph_method))
        # Create graph and load graph information
        # if training and hparams.makegraph:
        #     g = mesh2.GraphPyramid(heart_name, structure, num_meshfree, seq_len)
        #     g.make_graph()
        graphparams = net.get_graphparams(graph_dir, device, batch_size, heart_torso)

        # initialize datasets and dataloader
        train_dataset = HeartGraphDataset(root=root_dir, num_meshfree=num_meshfree, seq_len=seq_len,
                                        mesh_graph=graphparams["g"], mesh_graph_torso=graphparams["t_g"],
                                        heart_torso=heart_torso, train=False)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=training)
        
        corMfrees[heart_name] = train_dataset.getCorMfree()
        train_loaders[heart_name] = train_loader

        model.set_graphs(graphparams, heart_name)

    model.load_state_dict(torch.load(model_dir + '/' + hparams.vae_latest, map_location=device))
    model = model.eval().to(device)
    train.eval_real_new(model, train_loaders, exp_dir, corMfrees)


if __name__ == '__main__':
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # filename of the params
    fname_config = args.config + '.json'
    # read the params file
    json_path = osp.join(osp.dirname(osp.realpath('__file__')), "config", fname_config)
    net_path = osp.join(osp.dirname(osp.realpath('__file__')), 'net.py')
    hparams = utils.Params(json_path)
    torch.cuda.set_device(hparams.device)

    # check for a checkpoint passed in to resume from
    if args.checkpt != 'None':
        exp_path = 'experiments/{}/{}/{}'.format(hparams.model_type, args.config, args.checkpt)
        if os.path.isfile(exp_path):
            print("=> loading checkpoint '{}'".format(args.checkpt))
            checkpt = torch.load(exp_path, map_location=device)
            print('checkpoint: ', checkpt.keys())
            print("=> loaded checkpoint '{}' (epoch {})".format(args.checkpt, checkpt['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.checkpt))
            exit(0)
    else:
        checkpt = None

    # Go into the proper stage
    if args.stage == 1:  # generative modeling
        print('Stage 1: begin training vae for heart & torso ...')
        learn_vae_heart_torso(hparams, checkpt)
        print('Training vae completed!')
        print('--------------------------------------')
    elif args.stage == 2:
        print('Stage 2: begin evaluating vae for heart & torso ...')
        learn_vae_heart_torso(hparams, checkpt, training=False)
        print('Evaluating vae completed!')
        print('--------------------------------------')
    elif args.stage == 3:
        print('Stage 3: begin fine-tuning vae for heart & torso ...')
        learn_vae_heart_torso(hparams, checkpt, training=True, fine_tune=True)
        print('Tuning vae completed!')
        print('--------------------------------------')
    elif args.stage == 4:
        print('Stage 4: begin evaluating vae for heart & torso ...')
        real_data_new(hparams)
        print('Evaluating vae completed!')
        print('--------------------------------------')
    else:
        print('Invalid stage option!')
