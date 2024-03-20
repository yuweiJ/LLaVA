#!/bin/bash

export NCCL_P2P_DISABLE=1

CKPT_DIR=/mnt/intel/artifact_management/drive_vlm_dataset/checkpoints


deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path $CKPT_DIR/lmsys/vicuna-7b-v1.5 \
    --version v1 \
    --data_path ./playground/data/llava_v1_5_mix665k/llava_v1_5_mix665k.json \
    --image_folder ./playground/data/llava_v1_5_mix665k \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter $CKPT_DIR/llava-v1.5-7b/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio anyres \
    --image_grid_pinpoints "336, 672, 672, 336, 672, 672, 1008, 336" \
    --mm_patch_merge_type spatial_unpad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/exp/test_yw_llava-v1.5-7b_finetune \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
    # --image_grid_pinpoints "[[336, 672], [672, 336], [672, 672], [1008, 336], [336, 1008]]" \

    #--model_name_or_path $CKPT_DIR/lmsys/vicuna-7b-v1.5 \
    #--load_pt_cfg_only $CKPT_DIR/llava-v1.5-7b \
    #--load_pt_model_only $CKPT_DIR/lmsys/vicuna-7b-v1.5 \
