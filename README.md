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
With `nbody_fm` installed in edit mode (`-e` flag) its functionality can be imported as a python package, while also still editable in local storage. A working script showing its functionality can be found in `experiments/`:
```
cd experiments
python train_and_infer.py
```
