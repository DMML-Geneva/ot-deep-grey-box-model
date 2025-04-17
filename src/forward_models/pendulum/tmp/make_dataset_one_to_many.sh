#!/bin/bash

opt="--dt=0.05 --len-episode=50 --noise-std 1e-2 --library=scipy --outdir=./datasets/forward_models/pendulum/one_to_many/two_modes"
range_incomplete="--range-init -1.57 1.57 --range-omega 0.785 3.14"
range_complete="--range-init -1.57 1.57  
--range-omega 0.785 3.14 
--range-gamma 0 0.8 
--range-f 3.14 6.28
--range-A 0.0 40.0"
n_tr_incomplete_samples=1500 # 
n_tr_complete_samples=1500 # better to do a percentage of the labeled (incomplete samples) data
# One-to-many test samples
n_test_samples=1000

#conda activate ot-miss
# One-to-many Dataset problem configuration
## Train set
### Incomplete model - Labelled data
python -m src.forward_models.pendulum.generate_data ${opt} --n-samples ${n_tr_incomplete_samples} --seed=1234 ${range_incomplete}
## Complete model - Collocation points (unlabelled)
python -m src.forward_models.pendulum.generate_data ${opt} --n-samples ${n_tr_complete_samples} --seed=1234 ${range_complete}

## Test set
### Incomplete model - Labelled data
python -m src.forward_models.pendulum.generate_data ${opt} --n-samples ${n_test_samples} --seed=1 ${range_incomplete}
### Complete model, with n_stoch_samples per sample episode
python -m src.forward_models.pendulum.generate_data ${opt} --n-samples ${n_test_samples}  --seed=1 ${range_complete}