# CS3308 Machine Learning Project: Retrosynthetic Planning



## Setup

When establishing this repo, considering the large size of the data set, only the empty directory is reserved. When using this repo, data needs to be downloaded into the `data/` folder and run the code in `utils.py `for data preprocessing before further running other codes.



## Task1: Single-step retrosynthesis prediction

- Run SVM:
```shell
python train_svm.py
```
- Run MLP:

```shell
cd run/
python single_step_pred.py
```

- Run GLN:

~~~shell
cd gln/test
./test_single.sh schneider50k
~~~

## Task2: Molecule evaluation

- Run GAT:
~~~shell
cd run/
python molecule_eval.py
~~~

- Run MLP:

~~~shell
python train_mlp.py
~~~

## Task3: Multi-step retrosynthesis planning

set up the environment under the instructions of the README file in retro_star directory.

run Retro\*:

~~~sh
cd retro_star/retro_star/
python retro_plan.py --use_value_fn
~~~

