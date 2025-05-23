# 2025-DL Incremental Learning


A PyTorch Implementation of [iCaRL: Incremental Classifier and Representation Learning](https://arxiv.org/abs/1611.07725).


## requirement

python3.8.17
Pytorch1.3.0 
PyTorch 1.13.0 + CUDA 11.6
numpy

PIL

## run

```shell
python -u main.py
```

# Result
Resnet18+CIFAR100

| incremental step    | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9|
| ------------------- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| iCaRL test accuracy | 83.8|77.81|74.332|71.244|68.252|64.788|61.756|58.588|56.546|54.108|
| EWC test accuracy | 80.5|73.23|70.5|63.41|58.12|53.0|48.67|44.32|42.23|41.53|