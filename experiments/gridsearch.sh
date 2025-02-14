#! /bin/bash

# Gridsearch
# nepochs: 12, 20
# batch_size: 32, 64
# lr: 0.001, 0.0001
# feature_dim: 4
# beta: 0.5, 0.7

# Test
python train.py --nepochs 1 --batch_size 128 --lr 0.001 --feature_dim 4 --beta 0.5

# UDL1
# python train.py --nepochs 12 --batch_size 32 --lr 0.001 --feature_dim 4 --beta 0.5
# python train.py --nepochs 12 --batch_size 32 --lr 0.001 --feature_dim 4 --beta 0.7
# python train.py --nepochs 12 --batch_size 32 --lr 0.0001 --feature_dim 4 --beta 0.5
# python train.py --nepochs 12 --batch_size 32 --lr 0.0001 --feature_dim 4 --beta 0.7

# UDL2
# python train.py --nepochs 12 --batch_size 64 --lr 0.001 --feature_dim 4 --beta 0.5
# python train.py --nepochs 12 --batch_size 64 --lr 0.001 --feature_dim 4 --beta 0.7
# python train.py --nepochs 12 --batch_size 64 --lr 0.0001 --feature_dim 4 --beta 0.5
# python train.py --nepochs 12 --batch_size 64 --lr 0.0001 --feature_dim 4 --beta 0.7

# UDL3
# python train.py --nepochs 20 --batch_size 32 --lr 0.001 --feature_dim 4 --beta 0.5
# python train.py --nepochs 20 --batch_size 32 --lr 0.001 --feature_dim 4 --beta 0.7
# python train.py --nepochs 20 --batch_size 32 --lr 0.0001 --feature_dim 4 --beta 0.5
# python train.py --nepochs 20 --batch_size 32 --lr 0.0001 --feature_dim 4 --beta 0.7

# UDL4
# python train.py --nepochs 20 --batch_size 64 --lr 0.001 --feature_dim 4 --beta 0.5
# python train.py --nepochs 20 --batch_size 64 --lr 0.001 --feature_dim 4 --beta 0.7
# python train.py --nepochs 20 --batch_size 64 --lr 0.0001 --feature_dim 4 --beta 0.5
# python train.py --nepochs 20 --batch_size 64 --lr 0.0001 --feature_dim 4 --beta 0.7

touch models/keep.txt
touch losses/keep.txt
touch infer/keep.txt

cp models/* /data/UDL/models/
cp losses/* /data/UDL/losses/
cp infer/* /data/UDL/infer/
