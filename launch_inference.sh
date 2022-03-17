#!/bin/bash

python infer.py \
        --input_sources SI2017 ALTI \
        --target_source TLM5c \
        --batch_size 16 \
        --num_workers 2 \
        --save_hard \
        --save_soft \
        --save_error_map \
        --csv_fn /home/tanguyen/Documents/Projects/2020/ForestMapping/Code/ForestMapping/data/csv/SI2017_ALTI_TLM5c_val_viz.csv \
        --model_fn /home/tanguyen/Documents/Projects/2020/ForestMapping/Code/ForestMapping/output/baseline_hierarchical/training/baseline_hierarchical_model.pt \
        --output_dir /home/tanguyen/Documents/Projects/2020/ForestMapping/Code/ForestMapping/output/baseline_hierarchical/inference/epoch_1 \
        --evaluate \
        --overwrite \
        # > log.txt
        # --resume_training \
        # --debug \
        # --adapt_loss_weights \