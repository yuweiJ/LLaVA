#!/bin/bash

export NCCL_P2P_DISABLE=1

CKPT_DIR=/mnt/intel/artifact_management/drive_vlm_dataset/checkpoints
PLUS_DATA_DIR=/mnt/intel/artifact_management/drive_vlm_dataset/datasets/plus_ama_llava/finetune

exp_id=test_yw_finetune_v1_6_plus_pos_ego_$(date "+%Y%m%d-%H%M%S")
echo "$exp_id"


deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --load_pt_cfg_only $CKPT_DIR/llava-v1.6-vicuna-7b \
    --load_pt_model_only $CKPT_DIR/lmsys/vicuna-7b-v1.5 \
    --image_grid_pinpoints_str "[[336, 672], [672, 336], [672, 672], [1008, 336], [336, 1008]]" \
    --image_aspect_ratio anyres \
    --mm_patch_merge spatial_unpad \
    --version v1 \
    --data_path ./playground/data/test_first_100.json,$PLUS_DATA_DIR/pos_to_ego.json \
    --image_folder ./playground/data/llava_v1_5_mix665k,$PLUS_DATA_DIR \
    --vision_tower_name openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter $CKPT_DIR/llava-v1.5-7b/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir exp/$exp_id \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
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

    # --model_name_or_path $CKPT_DIR/llava-v1.5-7b \
    #--model_name_or_path $CKPT_DIR/lmsys/vicuna-7b-v1.5 \
    #--load_pt_cfg_only $CKPT_DIR/llava-v1.5-7b \
    #--load_pt_model_only $CKPT_DIR/lmsys/vicuna-7b-v1.5 \
    # --data_path ./playground/data/llava_v1_5_mix665k/llava_v1_5_mix665k.json \
    # --image_folder ./playground/data/llava_v1_5_mix665k \
    
