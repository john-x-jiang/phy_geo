import os
import numpy as np

idx = 38
loss_dir = './experiments/physics/p{}'.format(idx)
acc_phy_t = np.load(os.path.join(loss_dir, 'loss_phy_t.npy'))

loss_dir = './experiments/physics/p{}'.format(idx)
acc_phy_e = np.load(os.path.join(loss_dir, 'loss_phy_e.npy'))

loss_dir = './experiments/physics/p{}'.format(idx)
acc_hid_t = np.load(os.path.join(loss_dir, 'loss_hid_t.npy'))

loss_dir = './experiments/physics/p{}'.format(idx)
acc_hid_e = np.load(os.path.join(loss_dir, 'loss_hid_e.npy'))
import ipdb; ipdb.set_trace()
