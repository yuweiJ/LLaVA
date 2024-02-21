source set_env.sh

exp_id=test_yw_save_pt_v1_6_local_$(date "+%Y%m%d-%H%M%S")
echo "$exp_id"

export NCCL_P2P_DISABLE=1
CKPT_DIR=/mnt/intel/artifact_management/drive_vlm_dataset/checkpoints


deepspeed test/unit_tests/test_save_pt.py \
    --deepspeed ./scripts/zero2.json \
    --load_pt_cfg_only $CKPT_DIR/llava-v1.6-vicuna-7b \
    --load_pt_model_only $CKPT_DIR/lmsys/vicuna-7b-v1.5 \
    --version v1 \
    --data_path playground/data/single_test.json \
    --image_folder playground/data/llava_v1_5_mix665k \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter $CKPT_DIR/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --output_dir ./exp/$exp_id \
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
    --bf16 True \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing False \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    # --report_to wandb
    # --load_pt_model_only $CKPT_DIR/llava-v1.5-7b \

