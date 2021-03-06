#!/bin/bash

python src/train.py \
        --input_sources SI2017 \
        --target_source TLM5c \
        --batch_size 10 \
        --num_epochs 20 \
        --lr 1e-5 1e-6 1e-6 1e-7 \
        --learning_schedule 5 5 5 5 \
        --n_negative_samples 0 5 10 20 40 80 160 320 320 320 \
        --negative_sampling_schedule 2 2 2 2 2 2 2 2 2 2 \
        --decision h \
        --num_workers 2 \
        --no_user_input \
        --output_dir /media/data/charrez/multi-resolution-forest-mapping/results8 \
        --resume_training \
        # > log.txt
        # --debug \
        # --adapt_loss_weights \
        # --use_subset \
