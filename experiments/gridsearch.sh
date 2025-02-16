#! /bin/bash

# Gridsearch
# nepochs: 12, 20
# batch_size: 32, 64
# lr: 0.001, 0.0001
# feature_dim: 4
# beta: 0.5, 0.7

# Main train
python train.py --nepochs 70 --batch_size 32 --lr 0.0001 --feature_dim 4 --beta 0.5

python train.py --nepochs 12 --batch_size 32 --lr 0.001 --feature_dim 4 --beta 0.5
python train.py --nepochs 12 --batch_size 32 --lr 0.001 --feature_dim 4 --beta 0.7
python train.py --nepochs 12 --batch_size 32 --lr 0.0001 --feature_dim 4 --beta 0.5
python train.py --nepochs 12 --batch_size 32 --lr 0.0001 --feature_dim 4 --beta 0.7

python train.py --nepochs 12 --batch_size 64 --lr 0.001 --feature_dim 4 --beta 0.5
python train.py --nepochs 12 --batch_size 64 --lr 0.001 --feature_dim 4 --beta 0.7
python train.py --nepochs 12 --batch_size 64 --lr 0.0001 --feature_dim 4 --beta 0.5
python train.py --nepochs 12 --batch_size 64 --lr 0.0001 --feature_dim 4 --beta 0.7

python train.py --nepochs 20 --batch_size 32 --lr 0.001 --feature_dim 4 --beta 0.5
python train.py --nepochs 20 --batch_size 32 --lr 0.001 --feature_dim 4 --beta 0.7
python train.py --nepochs 20 --batch_size 32 --lr 0.0001 --feature_dim 4 --beta 0.5
python train.py --nepochs 20 --batch_size 32 --lr 0.0001 --feature_dim 4 --beta 0.7

python train.py --nepochs 20 --batch_size 64 --lr 0.001 --feature_dim 4 --beta 0.5
python train.py --nepochs 20 --batch_size 64 --lr 0.001 --feature_dim 4 --beta 0.7
python train.py --nepochs 20 --batch_size 64 --lr 0.0001 --feature_dim 4 --beta 0.5
python train.py --nepochs 20 --batch_size 64 --lr 0.0001 --feature_dim 4 --beta 0.7
