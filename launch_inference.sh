#!/bin/bash

python src/infer.py \
        --input_sources SI2017 ALTI \
        --target_source TLM5c \
        --batch_size 8 \
        --num_workers 2 \
        --save_hard \
        --save_soft \
        --csv_fn /media/data/charrez/multi-resolution-forest-mapping/src/data/csv/SI2017_ALTI_TLM5c_val_viz.csv \
        --model_fn /media/data/charrez/multi-resolution-forest-mapping/results/training/results_model.pt \
        --output_dir /media/data/charrez/multi-resolution-forest-mapping/results/baseline_hierarchical/inference/epoch_17 \
        --evaluate \
        --overwrite \
        # > log.txt
        # --save_error_map \
        # --model_fn /media/data/charrez/multi-resolution-forest-mapping/results/baseline_hierarchical/training/baseline_hierarchical_model.pt \
        # --resume_training \
        # --debug \
        # --adapt_loss_weights \