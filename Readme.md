# Running guidance

1. Check your cuda version. I've tested both `cu101` and `cu102`, and both of them are able to work.
2. Python version >= `3.8` would be recommended.
3. To install the python packages, change the variable `CUDA` in the script `req_torch_geo.sh`, then run it.
4. Check the configuration under `./config/`. Please use `seg04.json` for ODE-GCNN and `seg26.json` for ST-GCNN. Here are some important parameters:
   1. seq_len: the length of the input time sequence
   2. latent_dim: the dimension of the latent feature
   3. nf: the number of the feature in each layer
   4. anneal: when using stochastic model, this can be used as the parameter of the KLD term
   5. sample: the keeping rate of edges in a graph
   6. subset: the keeping rate of the input dataset
   7. smooth: the smoothing parameter of the regularization term
5. To train the model, run the following command:
   ```bash
   python main.py --config p06 --stage 1
   ```
6. To evaluate the model, run the following command:
   ```bash
   python main.py --config p06 --stage 2
   ```
