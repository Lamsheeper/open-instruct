export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

accelerate launch \
    open_instruct/finetune.py \
    --exp_name olmo2_7b_sft_original \
    --model_name_or_path allenai/OLMo-2-1124-7B \
    --tokenizer_name allenai/OLMo-2-1124-7B \
    --use_slow_tokenizer False \
    --add_bos \
    --use_flash_attn True \
    --dataset_mixer_list allenai/tulu-3-sft-olmo-2-mixture 1.0 \
    --max_seq_length 4096 \
    --preprocessing_num_workers 128 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1.0e-05 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --num_train_epochs 2 \
    --output_dir ./SFT_official_checkpoints/ \
    --logging_steps 100 \
    --reduce_loss sum \
    --gradient_checkpointing True \
    --checkpointing_steps 19500 \
    --dataset_mix_dir ./SFT_official_checkpoints/ \
    --hf_metadata_dataset allenai/olmo-instruct-evals \
    --keep_last_n_checkpoints 6 
