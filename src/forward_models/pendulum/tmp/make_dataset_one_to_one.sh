#!/bin/bash

opt="--dt=0.05 --len-episode=50 --noise-std 1e-2 --outdir=./datasets/forward_models/pendulum/one_to_one/"
range_complete="--range-init -1.57 1.57 --range-omega 0.785 3.14 --range-gamma 20 20 --range-f 6.28 6.28 --range-A 40 40.0"
range_incomplete="--range-init -1.57 1.57 --range-omega 0.785 3.14"
n_tr_incomplete_samples=10000
n_tr_complete_samples=10000 # better to do a percentage of the labeled (incomplete samples) data

# One-to-one dataset
one_to_one_test_samples=100000
one_to_one_seed=54321

#conda activate ot-miss

# One-to-one Dataset problem configuration
## Train set
## Incomplete model - Labelled data
#python -m src.forward_models.pendulum.generate_data ${opt} --n-samples ${n_tr_incomplete_samples} --seed=4321 ${range_incomplete}
##Complete model
#python -m src.forward_models.pendulum.generate_data ${opt} --n-samples ${n_tr_complete_samples} --seed=1234 --one-to-one ${one_to_one_seed} ${range_complete}

## Test set
### Incomplete model - Labelled data
python -m src.forward_models.pendulum.generate_data ${opt} --n-samples ${one_to_one_test_samples} --seed=1 ${range_incomplete}
#
### Complete model - Labelled data
#python -m src.forward_models.pendulum.generate_data ${opt} --n-samples ${one_to_one_test_samples} --seed=1 --one-to-one ${one_to_one_seed} ${range_complete}
