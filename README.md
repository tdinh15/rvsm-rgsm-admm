
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

For `alpha=0.5`:

|Method   |alpha   |num_bits   |Size (KB)   |Accuracy   |AUC   |AP   |f1   |
|---|---|---|---|---|---|---|---|
|baseline   |0.5   |n/a   |3224   |91.8   | 0.963 | 0.932 |    0.867 |
|PTQ   |0.5   |4   |372   |88.8   | 0.937 | 0.891 |    0.819 |
|QGT   |0.5   |4   |372   |91.5   | 0.960 | 0.928 |    0.864 |
|PTQ   |0.5   |2   |200   |68.4   | 0.637 | 0.393 |    0.552 |
|QGT   |0.5   |2   |200   |87.3   | 0.929 | 0.881 |    0.792 |


and for `alpha=0.25`:


|Method |num_bits   |Size (KB)   |Accuracy   |AUC   |AP   |f1   |
|---|---|---|---|---|---|---|---|
|baseline   |n/a   |1440   |90.4   | 0.952 | 0.925 |    0.843 |
|PTQ   |4   |133   |89.1   | 0.941 | 0.91  |    0.821 |
|QGT   |4   |133   |90.3   |0.952   |0.924   |0.838   |
|PTQ   |2   |81   |68.4   | 0.654 | 0.41  |    0.545 |
|QGT   |2   |81   |87.3   | 0.928 | 0.884 |    0.792 |
