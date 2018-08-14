# naf_examples
Examples of the ICML paper: [Neural Autoregressive Flows](http://proceedings.mlr.press/v80/huang18d.html)

To reproduce the experiments done in the paper, please refer to [this repo](https://github.com/CW-Huang/NAF).



* To use this repo, please also clone [this one](https://github.com/CW-Huang/torchkit) to the parent directory.

* The snippet in `example.py` can be summarized in three steps after choosing a data distribution the learn:

  1. initialize the model
  2. fitting the distribution
  3. visualize the learned model

For example:

```
denaf = naf.DensityEstimator(flowtype=1)
denaf.fit(distr, 2000)
fig = naf.visualize2D(distr, denaf, res=res, rng=rng)
```

For the swissroll distribution one would learn the following density model (right) using Deep Sigmoidal Flow, a neural transformer. The true data distribution is visualized on the left as a comparison. 

![alt text](https://github.com/CW-Huang/naf_examples/blob/master/figures/swiss_dsf.png "swissroll dsf")


Below is the result using affine transformer (aka the IAF by Kingma et al. 2016)

![alt text](https://github.com/CW-Huang/naf_examples/blob/master/figures/swiss_iaf.png "swissroll iaf")


