#!/bin/bash
python -m src.forward_models.advdiff.generate_data \
    --n-grids 20 --len-episode 50 \
    --dx 0.1 --dt 0.02 \
    --n-samples 250 \
    --n-stoch-samples 1 \
    --range-init-mag 0.5 1.5 \
    --range-dcoeff 1e-2 1e-1 \
    --range-ccoeff 1e-1 1e-1 \
    --method rk4 \
    --fixed_stoch_samples 0 \
    --outdir=./datasets/forward_models/advdiff/one_to_one/testing \
    --seed 13