import os
import torch
import transformers
from llava.model import *
from llava.model.multimodal_projector.builder import build_vision_projector

GPU_CKPT_DIR="/mnt/intel/artifact_management/drive_vlm_dataset/checkpoints"

def test_mm_projector():
    cfg_pt_path=os.path.join(GPU_CKPT_DIR, "llava-v1.6-vicuna-7b")
    model_pt_path=os.path.join(GPU_CKPT_DIR, "llava-v1.5-7b")
    config = LlavaConfig.from_pretrained(cfg_pt_path)
    print(config)
    model = LlavaLlamaForCausalLM.from_pretrained(
        model_pt_path,
        config=config,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    ).get_model()
    # mm_projector = build_vision_projector(config)
    print(model.mm_projector)
    for n, p in model.mm_projector.named_parameters():
        print(f"name={n}, param size={p.size()}")



if __name__ == "__main__":
    test_mm_projector()

