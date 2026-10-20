"""Extended schedule (PyTorch): GenoDirNet-SO, GenoDirNet-2 and the networks with one addition each.

Modules: features (one-hot input and the five ORF features of this schedule), sampling (fragment-aware windows,
error-aware noise, window-length bins), data (split files, data set, test windows) and models (sequence network
and its design space, sequence-ORF network, autoregressive models, likelihood-ratio features, pretraining heads,
length-weighted loss).
"""
