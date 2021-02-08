import pickle
import sys
import time
import os
import random

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy.io
import scipy.stats as stats
import torch
import torch.nn.functional as F

import utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def plot_losses(train_a, test_a, model_dir, num_epochs, loss_type):
    """Plot epoch against train loss and test loss 
    """
    # plot of the train/validation error against num_epochs
    fig, ax1 = plt.subplots(figsize=(6, 5))
    ax1.set_xticks(np.arange(0 + 1, num_epochs + 1, step=10))
    ax1.set_xlabel('epochs')
    ax1.plot(train_a, color='green', ls='-', label='train accuracy')
    ax1.plot(test_a, color='red', ls='-', label='test accuracy')
    h1, l1 = ax1.get_legend_handles_labels()
    ax1.legend(h1, l1, fontsize='14', frameon=False)
    ax1.grid(linestyle='--')
    plt.tight_layout()
    fig.savefig(model_dir + '/loss_{}.png'.format(loss_type), dpi=300, bbox_inches='tight', transparent=True)
    # plt.show()


def plot_reconstructions(model, x_loader, model_dir, corMfree, heart_name):
    """Plot inputs and reconstructions
    """
    batch_size = model.batch_size
    seq_len = model.seq_len
    model.eval()
    with torch.no_grad():
        data = next(iter(x_loader))
        heart_data, torso_data = data.y, data.x
        # torso_data = utils.norm_signal(torso_data)
        heart_data = heart_data.to(device)
        torso_data = torso_data.to(device)
        rtn, y_, l_h, mu, logvar, h, h_ = model(torso_data, heart_name)
        if type(rtn) == tuple:
            recon_data, recon_data_var = rtn
        else:
            recon_data = rtn

    N = 4  # change N to look at more samples
    # inds = np.random.permutation(seq_len)
    inds = [19, 20, 80, 85]
    x_sample = heart_data.view(batch_size, -1, seq_len)
    # x_sample = np.squeeze(x_sample.detach().cpu().numpy())
    x_sample = x_sample.detach().cpu().numpy()
    if batch_size > 1:
        x_sample = np.squeeze(x_sample[1, :, :])
    else:
        x_sample = x_sample[0, :, :]
    x_reconstruct = recon_data.view(batch_size, -1, seq_len)
    # x_reconstruct = np.squeeze(x_reconstruct.detach().cpu().numpy())  # encode then decode
    x_reconstruct = x_reconstruct.detach().cpu().numpy()
    if batch_size > 1:
        x_reconstruct = np.squeeze(x_reconstruct[1, :, :])
    else:
        x_reconstruct = x_reconstruct[0, :, :]

    if not os.path.exists(model_dir + '/pngs'):
        os.makedirs(model_dir + '/pngs')

    # plot the figure
    # x_sample_norm = utils.norm_signal(x_sample)
    # x_reconstruct_norm = utils.norm_signal(x_reconstruct)
    heart_cor, torso_cor = corMfree
    fig = plt.figure(figsize=(10, 5))
    for i in range(N):
        ax1 = fig.add_subplot(2, N, i + 1, projection='3d')
        p1 = ax1.scatter(heart_cor[:, 0], heart_cor[:, 1], heart_cor[:, 2], c=x_sample[:, inds[i]],
                        vmin=0.15, vmax=0.51, cmap=plt.cm.get_cmap('jet'))
        fig.colorbar(p1)
        ax2 = fig.add_subplot(2, N, N + i + 1, projection='3d')
        p2 = ax2.scatter(heart_cor[:, 0], heart_cor[:, 1], heart_cor[:, 2], vmin=0.15, vmax=0.51,
                        c=x_reconstruct[:, inds[i]], cmap=plt.cm.get_cmap('jet'))
        fig.colorbar(p2)

    plt.tight_layout()
    fig.savefig(model_dir + '/pngs/recons_{}.png'.format(heart_name), dpi=600, bbox_inches='tight', transparent=True)
    plt.close()

    # if not os.path.exists(model_dir + '/pngs/heart'):
    #     os.makedirs(model_dir + '/pngs/heart')
    # for i in range(101):
    #     fig = plt.figure(figsize=(3, 5))
    #     ax1 = fig.add_subplot(2, 1, 1, projection='3d')
    #     p1 = ax1.scatter(heart_cor[:, 0], heart_cor[:, 1], heart_cor[:, 2], c=x_sample[:, i],
    #                     vmin=0.15, vmax=0.51, cmap=plt.cm.get_cmap('jet'))
    #     fig.colorbar(p1)
    #     ax2 = fig.add_subplot(2, 1, 2, projection='3d')
    #     p2 = ax2.scatter(heart_cor[:, 0], heart_cor[:, 1], heart_cor[:, 2], vmin=0.15, vmax=0.51,
    #                     c=x_reconstruct[:, i], cmap=plt.cm.get_cmap('jet'))
    #     fig.colorbar(p2)
    #     plt.tight_layout()
    #     fig.savefig(model_dir + '/pngs/heart/{}_{}.png'.format(heart_name, i), dpi=600, bbox_inches='tight', transparent=True)
    #     plt.close()

    fig = plt.figure(figsize=(20, 5))
    t = np.arange(seq_len)
    inds = np.random.permutation(len(heart_cor))
    for i in range(N):
        ax1 = fig.add_subplot(2, N, i + 1)
        p1 = ax1.plot(t, x_sample[inds[i], :])
        ax2 = fig.add_subplot(2, N, N + i + 1)
        p2 = ax2.plot(t, x_reconstruct[inds[i], :])
    plt.tight_layout()
    fig.savefig(model_dir + '/pngs/TMP_{}.png'.format(heart_name), dpi=600, bbox_inches='tight', transparent=True)
    plt.close()


def plot_single_heart(model, x_loader, model_dir, corMfree, heart_name):
    if not os.path.exists(model_dir + '/pngs'):
        os.makedirs(model_dir + '/pngs')
    batch_size = model.batch_size
    seq_len = model.seq_len
    t = np.arange(seq_len)

    corts = []
    corss = []
    labels = []
    with torch.no_grad():
        for data in x_loader:
            heart_data, torso_data = data.y, data.x
            label = data.pos
            label = label.view(-1, 2)
            label = label.detach().cpu().numpy()
            label = np.squeeze(label)
            labels.append('{}_{}'.format(label[0], label[1]))
            # if label[0] in [1, 351, 501, 551, 601, 651, 1201, 1251, 1301]:
            #     continue
            heart_data = heart_data.to(device)
            torso_data = torso_data.to(device)
            rtn, y_, l_h, mu, logvar, h, h_ = model(torso_data, heart_name)
            if type(rtn) == tuple:
                recon_data, recon_data_var = rtn
            else:
                recon_data = rtn

            x_sample = heart_data.view(batch_size, -1, seq_len)
            x_sample = x_sample.detach().cpu().numpy()
            x_reconstruct = recon_data.view(batch_size, -1, seq_len)
            x_reconstruct = x_reconstruct.detach().cpu().numpy()
            _, num_nodes, _ = x_sample.shape

            # mask = utils.calc_corr_Pot_test(x_sample, x_reconstruct)
            # for j in range(num_nodes):
            #     if mask[j]:
            #         fig = plt.figure(figsize=(5, 5))
            #         ax1 = fig.add_subplot(211)
            #         p1 = ax1.plot(t, x_sample[0, j, :])
            #         ax2 = fig.add_subplot(212)
            #         p2 = ax2.plot(t, x_reconstruct[0, j, :])
            #         plt.tight_layout()
            #         fig.savefig(model_dir + '/pngs/{}_{}_{}_{}.png'.format(heart_name, label[0], label[1], j + 1), bbox_inches='tight', transparent=True)
            #         plt.close()

            cort = utils.calc_corr_Pot(x_sample, x_reconstruct)
            corts.append(cort)
            cors = utils.calc_corr_Pot_spatial(x_sample, x_reconstruct)
            corss.append(cors)

    corts = np.array(corts)
    corss = np.array(corss)
    # fig = plt.figure(1, figsize=(12, 8))
    # xaxis = np.arange(label.shape[0] / 16)
    # ax1 = fig.add_subplot(211)
    # ax1.plot(labels, corts, 'bo')
    # ax2 = fig.add_subplot(212)
    # ax2.plot(labels, corss, 'bo')
    # fig.savefig(model_dir + '/pngs/{}'.format(heart_name), bbox_inches='tight')
    # plt.close()
    print('heart: {}, cct = {:05.5f}, ccs = {:05.5f}'.format(heart_name, corts.mean(), corss.mean()))

    # with open(model_dir + '/pngs/{}.txt'.format(heart_name), 'a+') as f:
    #     for label, cors, cort in zip(labels, corss, corts):
    #         f.write('Label:{}, ccs = {:05.5f}, cct = {:05.5f}\n'.format(label, cors, cort))


def evaluation(model, x_loader, model_dir, corMfree, heart_name):
    batch_size = model.batch_size
    seq_len = model.seq_len
    n = len(x_loader.dataset)
    n = (n - n % batch_size)

    heart_cor, torso_cor = corMfree
    num_meshfree = len(heart_cor)
    num_torso = len(torso_cor)

    all_recons = np.empty((n, num_meshfree, seq_len))
    all_inps = np.empty((n, num_meshfree, seq_len))
    all_ys = np.empty((n, num_torso, seq_len))
    all_label = np.zeros((n, 2)).astype(int)
    mses, corrt, corrs, ats, dcs = [], [], [], [], []

    model.eval()
    i = 0
    with torch.no_grad():
        for data in x_loader:
            heart_data, torso_data = data.y, data.x
            label = data.pos
            label = label.view(-1, 2)
            torso_data = torso_data.to(device)
            rtn, y_, l_h, mu, logvar, h, h_ = model(torso_data, heart_name)
            if type(rtn) == tuple:
                recon_data, recon_data_var = rtn
            else:
                recon_data = rtn
            recon_data = recon_data.view(batch_size, -1, seq_len)
            recon_data = recon_data.detach().cpu().numpy()
            all_recons[i * batch_size:(i + 1) * batch_size, :, :] = recon_data

            x = heart_data.view(batch_size, -1, seq_len)
            x = x.detach().cpu().numpy()
            all_inps[i * batch_size:(i + 1) * batch_size, :, :] = x

            y_ = y_.view(batch_size, -1, seq_len)
            y_ = y_.detach().cpu().numpy()
            all_ys[i * batch_size:(i + 1) * batch_size, :, :] = y_

            label = label.detach().cpu().numpy()
            all_label[i, :] = label

            mses.append(utils.calc_msse(x, recon_data))
            corrt.append(utils.calc_corr_Pot(x, recon_data))
            corrs.append(utils.calc_corr_Pot_spatial(x, recon_data))
            # ats.append(utils.calc_AT(x, recon_data))
            # dc, _, _ = utils.calc_DC(x, recon_data)
            # dcs.append(dc)

            i += 1

    if not os.path.exists(model_dir + '/data'):
        os.makedirs(model_dir + '/data')
    scipy.io.savemat(model_dir + '/data/Tmp_{}.mat'.format(heart_name), {'U': all_recons, 'Y': all_ys, 'label': all_label})

    mse = utils.calc_msse(all_inps, all_recons)
    # at = utils.calc_AT(all_inps, all_recons)
    cort = utils.calc_corr_Pot(all_inps, all_recons)
    cors = utils.calc_corr_Pot_spatial(all_inps, all_recons)
    # dc, _, _ = utils.calc_DC(all_inps, all_recons)
    logs = 'Heart name: {}, mse: {:05.5f}, cort: {:05.5f}, cors: {:05.5f}'.format(heart_name, mse, cort, cors)
    print(logs)
    with open(os.path.join(model_dir, 'metric.txt'), 'a+') as f:
        f.write(logs + '\n')
    
    mses = np.array(mses)
    cort = np.array(corrt)
    cors = np.array(corrs)
    # atts = np.array(ats)
    # dcos = np.array(dcs)

    mse_mean = mses.mean()
    mse_std = mses.std()
    cort_mean = cort.mean()
    cort_std = cort.std()
    cors_mean = cors.mean()
    cors_std = cors.std()
    # at_mean = atts.mean()
    # at_std = atts.std()
    # dc_mean = dcos.mean()
    # dc_std = dcos.std()

    mean_stack = np.vstack((mse_mean, cort_mean, cors_mean))
    std_stack = np.vstack((mse_std, cort_std, cors_std))

    if not os.path.exists(model_dir + '/npys'):
        os.makedirs(model_dir + '/npys')
    
    np.save(os.path.join(model_dir, 'npys/mean_{}.npy'.format(heart_name)), mean_stack)
    np.save(os.path.join(model_dir, 'npys/std_{}.npy'.format(heart_name)), std_stack)

    metric_stack = np.vstack((mses, cors, cort))
    np.save(os.path.join(model_dir, 'npys/metric_{}.npy'.format(heart_name)), metric_stack)


def train(epoch, model, loss_function, phy_mode, smooth, hidden, optimizer, train_loaders, batch_size, seq_len, model_dir, anneal, sample=1):
    """Train a model and compute train loss
    """
    model.train()
    train_loss = 0
    bce_loss, kld_loss, phy_loss, smt_loss, hid_loss = 0, 0, 0, 0, 0
    n = 0  # len(train_dataset)
    heart_names = list(train_loaders.keys())
    random.shuffle(heart_names)
    for heart_name in heart_names:
        train_loader = train_loaders[heart_name]
        N = len(train_loader.dataset)
        for idx, data in enumerate(train_loader):
            optimizer.zero_grad()
            heart_data, torso_data = data.y, data.x
            
            heart_data = heart_data.to(device)
            torso_data = torso_data.to(device)
            rtn, y_, l_h, mu, logvar, h, h_ = model(torso_data, heart_name)
            if type(rtn) == tuple:
                mu_theta, logvar_theta = rtn
                logvar_theta = logvar_theta.view(-1, seq_len)
            else:
                mu_theta = rtn
            
            x = heart_data.view(-1, seq_len)
            mu_theta = mu_theta.view(-1, seq_len)
            y = torso_data.view(-1, seq_len)
            y_ = y_.view(-1, seq_len)
            # l_t = l_t.view(-1, seq_len)
            # l_h = l_h.view(-1, seq_len)
            if type(rtn) == tuple:
                loss, bce, kld = loss_function(mu_theta, logvar_theta, x, mu, logvar, batch_size, seq_len, epoch, anneal)
            else:
                loss, bce, kld, phy, smt, hid = loss_function(mu_theta, x, y_, y, l_h, mu, logvar, h, h_, phy_mode, smooth, hidden, batch_size, seq_len, epoch, anneal)
            loss.backward()
            train_loss += loss.item()
            bce_loss += bce.item()
            kld_loss += kld.item()
            phy_loss += phy.item()
            smt_loss += smt.item()
            hid_loss += hid.item()
            optimizer.step()
            n += 1
            utils.inline_print(f'Running epoch {epoch}, heart {heart_name}, batch {n}, Average loss for epoch: {str(train_loss / (n * batch_size))}')

            if idx > int(sample * N / batch_size):
                break


    train_loss /= (n * batch_size)
    bce_loss /= (n * batch_size)
    kld_loss /= (n * batch_size)
    phy_loss /= (n * batch_size)
    smt_loss /= (n * batch_size)
    hid_loss /= (n * batch_size)
    return train_loss, bce_loss, kld_loss, phy_loss, smt_loss, hid_loss


def test(epoch, model, loss_function, phy_mode, smooth, hidden, test_loaders, batch_size, seq_len, anneal):
    """Evaluated a trained model by computing validation loss
    """
    model.eval()
    test_loss = 0
    bce_loss, kld_loss, phy_loss, smt_loss, hid_loss = 0, 0, 0, 0, 0
    n = 0
    with torch.no_grad():
        for heart_name, test_loader in test_loaders.items():
            N = len(test_loader.dataset)
            for idx, data in enumerate(test_loader):
                heart_data, torso_data = data.y, data.x

                heart_data = heart_data.to(device)
                torso_data = torso_data.to(device)
                rtn, y_, l_h, mu, logvar, h, h_ = model(torso_data, heart_name)
                if type(rtn) == tuple:
                    mu_theta, logvar_theta = rtn
                    logvar_theta = logvar_theta.view(-1, seq_len)
                else:
                    mu_theta = rtn
                x = heart_data.view(-1, seq_len)
                mu_theta = mu_theta.view(-1, seq_len)
                y = torso_data.view(-1, seq_len)
                y_ = y_.view(-1, seq_len)
                # l_t = l_t.view(-1, seq_len)
                # l_h = l_h.view(-1, seq_len)
                if type(rtn) == tuple:
                    loss, bce, kld = loss_function(mu_theta, logvar_theta, x, mu, logvar, batch_size, seq_len, epoch, anneal)
                else:
                    loss, bce, kld, phy, smt, hid = loss_function(mu_theta, x, y_, y, l_h, mu, logvar, h, h_, phy_mode, smooth, hidden, batch_size, seq_len, epoch, anneal)
                test_loss += loss.item()
                bce_loss += bce.item()
                kld_loss += kld.item()
                phy_loss += phy.item()
                smt_loss += smt.item()
                hid_loss += hid.item()
                n += 1
                
                if idx > int(0.1 * N / batch_size):
                    break
    
    test_loss /= (n * batch_size)
    bce_loss /= (n * batch_size)
    kld_loss /= (n * batch_size)
    phy_loss /= (n * batch_size)
    smt_loss /= (n * batch_size)
    hid_loss /= (n * batch_size)
    return test_loss, bce_loss, kld_loss, phy_loss, smt_loss, hid_loss


def train_vae(model, checkpt, epoch_start, optimizer, lr_scheduler, train_loaders, test_loaders, loss_function, phy_mode, smooth,
              hidden, model_dir, num_epochs, batch_size, seq_len, corMfrees, anneal, sample=1):
    """
    """
    train_a = []
    bce_t, kld_t, phy_t, smt_t, hid_t = [], [], [], [], []

    test_a = []
    bce_e, kld_e, phy_e, smt_e, hid_e = [], [], [], [], []

    min_err = None

    # Load in current arrays if checkpt given
    if checkpt is not None:
        train_a, test_a = checkpt['train_acc'], checkpt['test_acc']

        bce_t, kld_t, phy_t, smt_t, hid_t = checkpt['bce_t'], checkpt['kld_t'], checkpt['phy_t'], \
                                            checkpt['smt_t'], checkpt['hid_t']

        bce_e, kld_e, phy_e, smt_e, hid_e = checkpt['bce_e'], checkpt['kld_e'], checkpt['phy_e'], \
                                            checkpt['smt_e'], checkpt['hid_e']

        min_err = checkpt['test_acc'][-1]

    for epoch in range(epoch_start, num_epochs + 1):
        # Train and test
        ts = time.time()
        train_acc, bce_acc_t, kld_acc_t, phy_acc_t, smt_acc_t, hid_acc_t = \
            train(epoch, model, loss_function, phy_mode, smooth, hidden, optimizer, train_loaders, batch_size, seq_len, model_dir, anneal, sample)
        test_acc, bce_acc_e, kld_acc_e, phy_acc_e, smt_acc_e, hid_acc_e = \
            test(epoch, model, loss_function, phy_mode, smooth, hidden, test_loaders, batch_size, seq_len, anneal)
        te = time.time()

        # Append epoch metrics to arrays
        train_a.append(train_acc)
        bce_t.append(bce_acc_t)
        kld_t.append(kld_acc_t)
        phy_t.append(phy_acc_t)
        smt_t.append(smt_acc_t)
        hid_t.append(hid_acc_t)
        test_a.append(test_acc)
        bce_e.append(bce_acc_e)
        kld_e.append(kld_acc_e)
        phy_e.append(phy_acc_e)
        smt_e.append(smt_acc_e)
        hid_e.append(hid_acc_e)

        # Step LR if 10th epoch
        # if epoch % 10 == 0:
        if lr_scheduler is not None:
            lr_scheduler.step(test_acc)
            last_lr = lr_scheduler._last_lr
        else:
            last_lr = 1

        # Generate the checkpoint for this current epoch
        checkpt = {
            # Base parameters to reload
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'cur_learning_rate': last_lr,
            'train_acc': train_a,
            'test_acc': test_a,

            # Holding the current arrays for training
            'bce_t': bce_t,
            'kld_t': kld_t,
            'phy_t': phy_t,
            'smt_t': smt_t,
            'hid_t': hid_t,

            # Holding the current arrays for testing
            'bce_e': bce_e,
            'kld_e': kld_e,
            'phy_e': phy_e,
            'smt_e': smt_e,
            'hid_e': hid_e,
        }

        # Check if current epoch is better than best so far
        if epoch == 1:
            min_err = test_acc
        else:
            if min_err > test_acc:
                min_err = test_acc
                torch.save(checkpt, model_dir + '/m_best')

        # Save the latest model
        torch.save(checkpt, model_dir + '/m_latest')

        # Every 10 epochs, reduce the learning rate and save that decade of training
        if epoch % 10 == 0:
            torch.save(checkpt, model_dir + '/m_' + str(epoch))

        # Print and write out epoch logs
        logs = 'Epoch: {:03d}, Time: {:.4f}, Train: {:.4f}, Test: {:.4f}                                             ' \
               ''.format(epoch, (te - ts) / 60, train_acc, test_acc)
        print(logs)
        with open(os.path.join(model_dir, 'log.txt'), 'a+') as f:
            f.write(logs + '\n')

    # Handles plotting and saving information over the training session
    plot_losses(train_a, test_a, model_dir, num_epochs, 'total')
    plot_losses(bce_t, bce_e, model_dir, num_epochs, 'bce')
    plot_losses(kld_t, kld_e, model_dir, num_epochs, 'kld')
    plot_losses(phy_t, phy_e, model_dir, num_epochs, 'phy')
    plot_losses(smt_t, smt_e, model_dir, num_epochs, 'smt')
    plot_losses(hid_t, hid_e, model_dir, num_epochs, 'hid')
    train_a = np.array(train_a)
    test_a = np.array(test_a)
    bce_t = np.array(bce_t)
    bce_e = np.array(bce_e)
    kld_t = np.array(kld_t)
    kld_e = np.array(kld_e)
    phy_t = np.array(phy_t)
    phy_e = np.array(phy_e)
    smt_t = np.array(smt_t)
    smt_e = np.array(smt_e)
    hid_t = np.array(hid_t)
    hid_e = np.array(hid_e)
    np.save(os.path.join(model_dir, 'loss_train.npy'), train_a)
    np.save(os.path.join(model_dir, 'loss_test.npy'), test_a)
    np.save(os.path.join(model_dir, 'loss_bce_t.npy'), bce_t)
    np.save(os.path.join(model_dir, 'loss_bce_e.npy'), bce_e)
    np.save(os.path.join(model_dir, 'loss_kld_t.npy'), kld_t)
    np.save(os.path.join(model_dir, 'loss_kld_e.npy'), kld_e)
    np.save(os.path.join(model_dir, 'loss_phy_t.npy'), phy_t)
    np.save(os.path.join(model_dir, 'loss_phy_e.npy'), phy_e)
    np.save(os.path.join(model_dir, 'loss_smt_t.npy'), smt_t)
    np.save(os.path.join(model_dir, 'loss_smt_e.npy'), smt_e)
    np.save(os.path.join(model_dir, 'loss_hid_t.npy'), hid_t)
    np.save(os.path.join(model_dir, 'loss_hid_e.npy'), hid_e)
    # for heart_name, corMfree in corMfrees.items():
    #     plot_zmean(model, test_loaders[heart_name], model_dir, corMfree, heart_name)
    #     plot_reconstructions(model, test_loaders[heart_name], model_dir, corMfree, heart_name)


def get_network_paramcount(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = sum([np.prod(p.size()) for p in model_parameters])
    return num_params

    
def eval_vae(model, test_loaders, model_dir, 
             batch_size, seq_len, corMfrees):
    """
    """
    # 
    num_params = get_network_paramcount(model)
    print('The number of network prameters: {}\n'.format(num_params))
    for heart_name, corMfree in corMfrees.items():
        evaluation(model, test_loaders[heart_name], model_dir, corMfree, heart_name)


def returnScar(u, seq_len):
    m, n = u.shape
    if n == seq_len:
        u = u.transpose()
        T = n
    elif m == seq_len:
        T = m
    else:
        print('Dimension mismatch!!')

    u_apd = np.sum((u > 0.75), axis=0)
    u_scar = u_apd > (0.25 * T)

    return u_scar.astype(float)


def eval_real(model, Y, heart_name, model_dir, heart_cor):
    batch_size = model.batch_size
    seq_len = model.seq_len

    Y = torch.Tensor(Y).to(device)
    Y = Y.view(batch_size, -1, seq_len)
    rtn, y_, l_h, mu, logvar, h, h_ = model(torso_data, heart_name)
    if type(rtn) == tuple:
        recon_data, recon_data_var = rtn
    else:
        recon_data = rtn
    recon_data = recon_data.view(batch_size, -1, seq_len)
    recon_data = recon_data.detach().cpu().numpy()
    recon_data = np.squeeze(recon_data)

    # for i in range(101):
    #     fig = plt.figure(figsize=(3, 3))
    #     ax1 = fig.add_subplot(1, 1, 1, projection='3d')
    #     p1 = ax1.scatter(heart_cor[:, 0], heart_cor[:, 1], heart_cor[:, 2], c=recon_data[:, i],
    #                     vmin=0.15, vmax=0.51, cmap=plt.cm.get_cmap('jet'))
    #     fig.colorbar(p1)
    #     plt.tight_layout()
    #     fig.savefig(model_dir + '/heart/{}_{}.png'.format(heart_name, i), dpi=600, bbox_inches='tight', transparent=True)
    #     plt.close()
    if not os.path.exists(model_dir + '/real_data'):
        os.makedirs(model_dir + '/real_data')
    scipy.io.savemat(model_dir + '/real_data/{}.mat'.format(heart_name), {'U': recon_data})
    scipy.io.savemat(model_dir + '/real_data/{}_scar.mat'.format(heart_name), {'U': returnScar(recon_data, seq_len)})


def activation_time(u, M):
    # import ipdb; ipdb.set_trace()
    u = u.transpose(2, 0, 1)
    M = M.transpose(2, 0, 1)
    w, m, n = u.shape
    # prediction
    u_new = np.roll(u, -1, axis=0)
    u_slope = (u_new - u)[:-1, :, :]
    u_AT = np.argmax(u_slope, axis=0)

    # gt
    M_new = np.roll(M, -1, axis=0)
    M_slope = (M_new - M)[:-1, :, :]
    M_AT = np.argmin(M_slope, axis=0)

    corr_coeff = 0
    for i in range(m):
        true_AT = u_AT[i, :]
        x_AT = M_AT[i, :]
        corr_coeff = corr_coeff + stats.pearsonr(true_AT, x_AT)[0]
    return corr_coeff / m


def pacing_site(u, cor, runs):
    u = u.transpose(2, 0, 1)
    w, m, n = u.shape
    # gt
    pacing_sites = {
        7: np.array([-13.5965796645031, 7.24232566240460, 288.740128516268]),
        21: np.array([-22.2311363379111, -12.9739238101521, 259.167604248245]),
        27: np.array([-19.6819057687105, -10.3206092363913, 270.089603294572]),
        36: np.array([-2.04549732629463, 8.47844433358722, 267.852802702838]),
        45: np.array([-47.2671985410967, 22.1954331396453, 252.611483297335])
    }

    # prediction
    u_new = np.roll(u, -1, axis=0)
    u_slope = (u_new - u)[:-1, :, :]
    u_AT = np.argmax(u_slope, axis=0)

    u_AT = np.squeeze(u_AT)
    selected_num = 30
    idx = np.argpartition(u_AT, selected_num)[:selected_num]
    selected_cor = cor[idx, :]
    p_site_pred = np.sum(selected_cor, axis=0) / selected_num

    dist = p_site_pred - pacing_sites[runs]
    dist = np.sqrt(np.sum(dist.T * dist))
    return dist, p_site_pred


def eval_real_new(model, data_loaders, model_dir, corMfrees):
    batch_size = model.batch_size
    seq_len = model.seq_len
    model.eval()

    for heart_name, corMfree in corMfrees.items():
        x_loader = data_loaders[heart_name]
        n = len(x_loader.dataset)
        n = (n - n % batch_size)
        heart_cor, torso_cor = corMfree
        num_meshfree = len(heart_cor)

        all_recons = np.empty((n, num_meshfree, seq_len))
        if not os.path.exists(model_dir + '/data_directapply'):
            os.makedirs(model_dir + '/data_directapply')

        with torch.no_grad():
            for data in x_loader:
                heart_data, torso_data = data.y, data.x
                label = data.pos
                label = label.view(-1, 2)
                label = label.detach().cpu().numpy()
                label = np.squeeze(label)
                heart_data = heart_data.to(device) * 1e-2
                torso_data = torso_data.to(device) * 1e-2
                rtn, y_, l_h, mu, logvar, h, h_ = model(torso_data, heart_name)
                if type(rtn) == tuple:
                    recon_data, recon_data_var = rtn
                else:
                    recon_data = rtn
                recon_data = recon_data.view(batch_size, -1, seq_len)
                recon_data = recon_data.detach().cpu().numpy()
                x = heart_data.view(batch_size, -1, seq_len)
                x = x.detach().cpu().numpy()
                if label[1] == 55:
                    continue
                err, p_site_pred = pacing_site(x, heart_cor, label[1])
                logs = 'Beat {}, Run {:02d}, err: {:05.5f}'.format(label[0], label[1], err)
                print(logs)
                with open(os.path.join(model_dir, 'log_err.txt'), 'a+') as f:
                    f.write(logs + '\n')
                cors = '{}, {}, {}'.format(p_site_pred[0], p_site_pred[1], p_site_pred[2])
                # print(cors)
                with open(os.path.join(model_dir, 'log_pacing.txt'), 'a+') as f:
                    f.write(cors + '\n')
                # at = activation_time(recon_data, x)
                # logs = 'Beat {}, Run {:02d}, at: {:05.5f}'.format(label[0], label[1], at)
                # print(logs)
                # with open(os.path.join(model_dir, 'log_directapply.txt'), 'a+') as f:
                #     f.write(logs + '\n')
                # scipy.io.savemat(model_dir + '/data_directapply/Tmp_{}_{}.mat'.format(label[0], label[1]), {'U': recon_data})
