#!/bin/bash

python src/infer.py \
        --input_sources SI1946 SI2017\
        --target_source TLM5c \
        --batch_size 8 \
        --num_workers 2 \
        --save_hard \
        --save_soft \
        --csv_fn /media/data/charrez/multi-resolution-forest-mapping/src/data/csv/SI1946_SI2017_TLM5c_val.csv \
        --model_fn /media/data/charrez/multi-resolution-forest-mapping/results3/training/results3_model.pt \
        --output_dir /media/data/charrez/multi-resolution-forest-mapping/results3/inference/val \
        --evaluate \
        --overwrite \
        --compare_dates \
        # > log.txt
        # --save_error_map \
        # --debug \
        # --adapt_loss_weights \