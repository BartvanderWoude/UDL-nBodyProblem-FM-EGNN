#! /bin/bash

# Gridsearch
# nepochs: 20, 40
# batch_size: 64, 128
# lr: 0.001, 0.0001
# feature_dim: 3, 5
# beta: 0.5, 0.7

python train.py --nepochs 20 --batch_size 64 --lr 0.001 --feature_dim 3 --beta 0.5
python train.py --nepochs 20 --batch_size 64 --lr 0.001 --feature_dim 3 --beta 0.7
python train.py --nepochs 20 --batch_size 64 --lr 0.001 --feature_dim 5 --beta 0.5
python train.py --nepochs 20 --batch_size 64 --lr 0.001 --feature_dim 5 --beta 0.7
python train.py --nepochs 20 --batch_size 64 --lr 0.0001 --feature_dim 3 --beta 0.5
python train.py --nepochs 20 --batch_size 64 --lr 0.0001 --feature_dim 3 --beta 0.7
python train.py --nepochs 20 --batch_size 64 --lr 0.0001 --feature_dim 5 --beta 0.5
python train.py --nepochs 20 --batch_size 64 --lr 0.0001 --feature_dim 5 --beta 0.7

python train.py --nepochs 20 --batch_size 128 --lr 0.001 --feature_dim 3 --beta 0.5
python train.py --nepochs 20 --batch_size 128 --lr 0.001 --feature_dim 3 --beta 0.7
python train.py --nepochs 20 --batch_size 128 --lr 0.001 --feature_dim 5 --beta 0.5
python train.py --nepochs 20 --batch_size 128 --lr 0.001 --feature_dim 5 --beta 0.7
python train.py --nepochs 20 --batch_size 128 --lr 0.0001 --feature_dim 3 --beta 0.5
python train.py --nepochs 20 --batch_size 128 --lr 0.0001 --feature_dim 3 --beta 0.7
python train.py --nepochs 20 --batch_size 128 --lr 0.0001 --feature_dim 5 --beta 0.5
python train.py --nepochs 20 --batch_size 128 --lr 0.0001 --feature_dim 5 --beta 0.7

python train.py --nepochs 40 --batch_size 64 --lr 0.001 --feature_dim 3 --beta 0.5
python train.py --nepochs 40 --batch_size 64 --lr 0.001 --feature_dim 3 --beta 0.7
python train.py --nepochs 40 --batch_size 64 --lr 0.001 --feature_dim 5 --beta 0.5
python train.py --nepochs 40 --batch_size 64 --lr 0.001 --feature_dim 5 --beta 0.7
python train.py --nepochs 40 --batch_size 64 --lr 0.0001 --feature_dim 3 --beta 0.5
python train.py --nepochs 40 --batch_size 64 --lr 0.0001 --feature_dim 3 --beta 0.7
python train.py --nepochs 40 --batch_size 64 --lr 0.0001 --feature_dim 5 --beta 0.5
python train.py --nepochs 40 --batch_size 64 --lr 0.0001 --feature_dim 5 --beta 0.7

python train.py --nepochs 40 --batch_size 128 --lr 0.001 --feature_dim 3 --beta 0.5
python train.py --nepochs 40 --batch_size 128 --lr 0.001 --feature_dim 3 --beta 0.7
python train.py --nepochs 40 --batch_size 128 --lr 0.001 --feature_dim 5 --beta 0.5
python train.py --nepochs 40 --batch_size 128 --lr 0.001 --feature_dim 5 --beta 0.7
python train.py --nepochs 40 --batch_size 128 --lr 0.0001 --feature_dim 3 --beta 0.5
python train.py --nepochs 40 --batch_size 128 --lr 0.0001 --feature_dim 3 --beta 0.7
python train.py --nepochs 40 --batch_size 128 --lr 0.0001 --feature_dim 5 --beta 0.5
python train.py --nepochs 40 --batch_size 128 --lr 0.0001 --feature_dim 5 --beta 0.7
