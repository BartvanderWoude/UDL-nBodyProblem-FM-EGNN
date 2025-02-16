# EGNN with FM for 3-body problem
## Setup
Create a python virtual environment and activate it.

The main dependencies for this code are `egnn_pytorch` and `flow_matching`.
Install the package `flow-matching` with module `flow_matching`:
```
pip install flow-matching
```

Install package `egnn-pytorch` with module `egnn_pytorch` by moving to its source folder and running:
```
python setup.py install
```

Lastly, in the source folder of this repo we install the accompanying module `nbody_fm`:
```
pip install -e .
```

## Usage
With `nbody_fm` installed in edit mode (`-e` flag) its functionality can be imported as a python package, while also still editable in local storage.

Training and inference can be found in `experiments/`. There are shell scripts to run the full gridsearch and full inference:
```
cd experiments
./gridsearch.sh
./inference.sh
```

However, `train.py` and `infer.py` in `experiments/` can also be called from the command-line using CLI arguments.
```
python train.py --nepochs 20 --batch_size 32 --lr 0.001 --feature_dim 4 --beta 0.5
python infer.py --model best_model_N70_batch32_lr0.0001_fd4_b0.5.pth --infermethod euler --solverstepsize 0.01 --lookahead 10 --inferencesteps 500 [--fulldata]
```

## Evaluation
Aside from the code described in the previous section, the data folder contains the results from our gridsearch, training and inference. `Three_body_2D_test.m` generates our dataset `3body_2d_data.csv`, `make_infer_table.py` and `make_loss_table.py` generate latex tables used in the report, `plot_losses.py` creates loss plots, `make_plots.m` with function `plot_inferred.m` create the inference plots used in the report. The folders `infer/`, `losses/` and `models/` contain our results.

The folder `data_additional_experiment/` contains the results from the additional experiment described in the report in addition to the fine-tuned loss scripts. Please note that the model described in `additional experiment` can be found in the file `additionalexperiment_model.py` in `nbody_fm`. In order to use this model instead of the base model, swap out the aforementioned file with `model.py`.