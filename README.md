# The SA-GNN model for Sequential Recommendation
The implementation of the paper:

*Yansen Zhang, Chenhao Hu, Genan Dai, Weiyang Kong, and Yubao Liu, "**Self-Adaptive Graph Neural Networks for Personalized Sequential Recommendation**", in the 28th International Conference on Neural Information Processing (ICONIP) (**ICONIP 2021**)*


## Environments

- python 3.6.8
- PyTorch (version: 1.6.0)
- GPU (GeForce RTX 2080 Ti)

## Dataset

The datasets and data preprocessing can refer to [HGN](https://github.com/allenjack/HGN).

## Example to run the code


Train and evaluate the model:

```
python run.py
```

## Some Baselines

* [MA-GNN] can be found in my another repos
* [-GAT] can be found in the file "gating_network_gat.py"

## Comments

You can change the $L$ and $T$ according to your needs, but code should be adapted too

## Acknowledgements

I found these repos useful (while developing this one):

* [official GAT](https://github.com/PetarV-/GAT)
* [HGN](https://github.com/allenjack/HGN)


## Issues/Pull Requests/Feedbacks

**Feel free to send me an email or add issues if you have any questions.**

**Please cite our paper if you use our code. Thanks!**

Author: Yansen Zhang (zhangys7@mail2.sysu.edu.cn)
