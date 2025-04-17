#!/bin/bash

opt="--dt=0.05 --len-episode=200 --noise-std 1e-2 --outdir=./datasets/forward_models/pendulum/one_to_many/two_modes"
range_incomplete="--range-init -1.57 1.57 --range-omega 0.785 3.14"
range_complete="--range-init -1.57 1.57  
--range-omega 0.785 3.14 
--range-gamma 0.1 20 --gamma-dist bernoulli 
--range-f 4.0 6.28 --f-dist bernoulli  
--range-A 10 40.0 --A-dist bernoulli  
--single-sample-parameters 1"
n_tr_incomplete_samples=500000 # 
n_tr_complete_samples=3000 # better to do a percentage of the labeled (incomplete samples) data
# One-to-many test samples
n_test_samples=3000
n_stoch_samples=6


#conda activate ot-miss

# One-to-many Dataset problem configuration
## Train set
### Incomplete model - Labelled data
python -m src.forward_models.pendulum.generate_data ${opt} --n-samples ${n_tr_incomplete_samples} --seed=4321 ${range_incomplete}
## Complete model - Collocation points (unlabelled)
#python -m src.forward_models.pendulum.generate_data ${opt} --n-samples ${n_tr_complete_samples} --n-stoch-samples ${n_stoch_samples} --seed=1234 ${range_complete}

## Test set
### Incomplete model - Labelled data
python -m src.forward_models.pendulum.generate_data ${opt} --n-samples ${n_test_samples} --seed=1 ${range_incomplete}
### Complete model, with n_stoch_samples per sample episode
#python -m src.forward_models.pendulum.generate_data ${opt} --n-samples ${n_test_samples} --n-stoch-samples ${n_stoch_samples} --seed=1 ${range_complete}