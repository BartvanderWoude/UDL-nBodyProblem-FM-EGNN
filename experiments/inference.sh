#! /bin/bash

# Best model: N20_batch32_lr0.0001_fd4_b0.5

# Whole dataset
# No reset
python infer.py --model best_model_N70_batch32_lr0.0001_fd4_b0.5.pth --infermethod euler --solverstepsize 0.01 --lookahead 25000 --inferencesteps 25000 --fulldata True
python infer.py --model best_model_N70_batch32_lr0.0001_fd4_b0.5.pth --infermethod dopri5 --lookahead 25000 --inferencesteps 25000 --fulldata True

# Reset every 1 step, maximum 12000 steps
python infer.py --model best_model_N70_batch32_lr0.0001_fd4_b0.5.pth --infermethod euler --solverstepsize 0.01 --lookahead 1 --inferencesteps 12000 --fulldata True
python infer.py --model best_model_N70_batch32_lr0.0001_fd4_b0.5.pth --infermethod dopri5 --lookahead 1 --inferencesteps 12000 --fulldata True

# Reset every 10 steps, maximum 12000 steps
python infer.py --model best_model_N70_batch32_lr0.0001_fd4_b0.5.pth --infermethod euler --solverstepsize 0.01 --lookahead 10 --inferencesteps 12000 --fulldata True
python infer.py --model best_model_N70_batch32_lr0.0001_fd4_b0.5.pth --infermethod dopri5 --lookahead 10 --inferencesteps 12000 --fulldata True

# Reset every 100 steps, maximum 12000 steps
python infer.py --model best_model_N70_batch32_lr0.0001_fd4_b0.5.pth --infermethod euler --solverstepsize 0.01 --lookahead 100 --inferencesteps 12000 --fulldata True
python infer.py --model best_model_N70_batch32_lr0.0001_fd4_b0.5.pth --infermethod dopri5 --lookahead 100 --inferencesteps 12000 --fulldata True

# Test dataset
# No reset
python infer.py --model best_model_N70_batch32_lr0.0001_fd4_b0.5.pth --infermethod euler --solverstepsize 0.01 --lookahead 25000 --inferencesteps 25000 --fulldata False
python infer.py --model best_model_N70_batch32_lr0.0001_fd4_b0.5.pth --infermethod dopri5 --lookahead 25000 --inferencesteps 25000 --fulldata False

# Reset every 1 step
python infer.py --model best_model_N70_batch32_lr0.0001_fd4_b0.5.pth --infermethod euler --solverstepsize 0.01 --lookahead 1 --inferencesteps 12000 --fulldata False
python infer.py --model best_model_N70_batch32_lr0.0001_fd4_b0.5.pth --infermethod dopri5 --lookahead 1 --inferencesteps 12000 --fulldata False

# Reset every 10 steps
python infer.py --model best_model_N70_batch32_lr0.0001_fd4_b0.5.pth --infermethod euler --solverstepsize 0.01 --lookahead 10 --inferencesteps 12000 --fulldata False
python infer.py --model best_model_N70_batch32_lr0.0001_fd4_b0.5.pth --infermethod dopri5 --lookahead 10 --inferencesteps 12000 --fulldata False

# Reset every 100 steps
python infer.py --model best_model_N70_batch32_lr0.0001_fd4_b0.5.pth --infermethod euler --solverstepsize 0.01 --lookahead 100 --inferencesteps 12000 --fulldata False
python infer.py --model best_model_N70_batch32_lr0.0001_fd4_b0.5.pth --infermethod dopri5 --lookahead 100 --inferencesteps 12000 --fulldata False

touch models/keep.txt
touch losses/keep.txt
touch infer/keep.txt

# # cp models/* /data/UDL/models/
# # cp losses/* /data/UDL/losses/
cp infer/* /data/UDL/infer/
