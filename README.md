
# RVSM/RGSM for weights and channel pruning
Prune DNN's using Relaxed Variables Splitting Method (RVSM) and Relaxed Group-wise Splitting Method (RGSM). A comparison with the Alternating Direction Method of Multipliers (ADMM) is also available.

#### RVSM is used for weight (unstructured) pruning and was proposed in:
T. Dinh, J. Xin, “Convergence of a relaxed variable splitting method for learning sparse neural networks via $\ell_1, \ell_0$, and transformed-$\ell_1$ penalties”, arXiv preprint arXiv: [https://arxiv.org/abs/1901.09731](https://arxiv.org/abs/1901.09731) 

#### RGSM is used for channel (structured) pruning and was proposed in:
B. Yang, J. Lyu, S. Zhang, Y-Y Qi, J. Xin “Channel Pruning for Deep Neural Networks via a Relaxed Group-wise Splitting Method”, In Proc. of 2nd International Conference on AI for Industries, Laguna Hills, CA, Sept. 25-27, 2019

#### Instruction:
The available models are in the models folder and can be specified in main.py. Run main.py to train the model, specify the method and corresponding parameters. The available training options are: 

default: standard SGD

rvsm: hyper-parameters are --beta and --lamb, default values are [1e-2, 1e-6]

rgsm: hyper-parameters are --beta1, --lamb1, and --lamb2, default value are [1, 1e-2, 1e-5]

admm: hyper-parameters are --pcen and --sparsity ('elem' or 'channel'), default values are [80, elem]


```
python main.py --method default
python main.py --method rvsm --beta 5e-2 --lamb 1e-5
python main.py --method rgsm --beta1 1 --lamb1 5e-3 --lamb2 2e-5
python main.py --method admm --pcen 60 --sparsity channel
```

## BLOG!!!
https://medium.com/@tdinh15/80-days-to-get-into-faang-5c77f27d5224