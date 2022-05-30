#!/bin/bash

python src/infer.py \
        --input_sources SI1946 SI2017\
        --target_source TLM5c \
        --batch_size 8 \
        --num_workers 2 \
        --save_hard \
        --save_soft \
        --model_fn /media/data/charrez/multi-resolution-forest-mapping/results6/training/results6_model.pt \
        --csv_fn /media/data/charrez/multi-resolution-forest-mapping/src/data/csv/SI1946_SI2017_TLM5c_test_unchanged.csv \
        --output_dir /media/data/charrez/multi-resolution-forest-mapping/results6/inference/unchanged \
        --evaluate \
        --overwrite \
        --compare_dates \
        # --csv_fn /media/data/charrez/multi-resolution-forest-mapping/src/data/csv/SI1946_SI2017_TLM5c_test_unchanged.csv \
        # --output_dir /media/data/charrez/multi-resolution-forest-mapping/results4/inference/unchanged \
        # --csv_fn /media/data/charrez/multi-resolution-forest-mapping/src/data/csv/SI1946_SI2017_TLM5c_test_random.csv \
        # --output_dir /media/data/charrez/multi-resolution-forest-mapping/results4/inference/random \
        # --csv_fn /media/data/charrez/multi-resolution-forest-mapping/src/data/csv/SI1946_SI2017_TLM5c_patch_analysis.csv \
        # --output_dir /media/data/charrez/multi-resolution-forest-mapping/results4/inference/patch_analysis \
        # > log.txt
        # --save_error_map \
        # --debug \
        # --adapt_loss_weights \