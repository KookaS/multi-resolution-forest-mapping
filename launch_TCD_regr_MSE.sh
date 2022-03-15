#!/bin/bash

python src/train.py \
        --input_sources IMAGE2017 \
        --target_source TLM5c \
        --aux_target_sources VHM TCD1 \
        --batch_size 16 \
        --num_epochs 5 \
        --lr 1e-5 \
        --learning_schedule 5 \
        --lambda_regr 1 \
        --lambda_corr 0 \
        --n_negative_samples 10 \
        --negative_sampling_schedule 5 \
        --decision f \
        --penalize_residual \
        --regression_loss MSE \
        --num_workers 2 \
        --resume_training \
        --output_dir /home/tanguyen/Documents/Projects/2020/ForestMapping/Code/ForestMapping/output/TCD1_regr_only_MSE_full_ds \
        > results/logMSE.txt #2>&1
        # --use_subset \
        # --aux_target_sources \
        # --debug \
        
        # --adapt_loss_weights \