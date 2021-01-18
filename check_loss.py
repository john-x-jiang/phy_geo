import os
import numpy as np

loss_dir = './experiments/physics/p00'
p00 = np.load(os.path.join(loss_dir, 'loss_phy_t.npy'))

loss_dir = './experiments/physics/p01'
p01 = np.load(os.path.join(loss_dir, 'loss_phy_t.npy'))
import ipdb; ipdb.set_trace()
