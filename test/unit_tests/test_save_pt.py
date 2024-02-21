import torch
import transformers
from llava.model import *
from llava import conversation as conversation_lib
# from llava.model.multimodal_projector.builder import build_vision_projector
from llava.train.train import (ModelArguments, DataArguments, TrainingArguments,
                               make_supervised_data_module, safe_save_model_for_hf_trainer)
from llava.train.llava_trainer import LLaVATrainer


def test_save_pt():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    config = LlavaConfig.from_pretrained(model_args.load_pt_cfg_only,
                                         exponential_decay_length_penalty=None)
    print(f"BEFORE TRAIN: pretrained config do_sample={config.do_sample}, temperature={config.temperature}")

    # print(config)
    model = LlavaLlamaForCausalLM.from_pretrained(
        model_args.load_pt_model_only,
        config=config,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.load_pt_cfg_only,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    version = "v1"
    if version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    vision_tower = "openai/clip-vit-large-patch14-336"
    model.get_model().initialize_vision_modules(
        model_args=model_args,
        fsdp=training_args.fsdp
    )

    vision_tower = model.get_vision_tower()
    vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

    data_args.image_processor = vision_tower.image_processor
    data_args.is_multimodal = True

    model.config.image_aspect_ratio = data_args.image_aspect_ratio
    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length

    model.config.use_cache = False
    model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
    training_args.use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
    model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    # for n, p in model.named_parameters():
    #     print(f"name={n}, requires_grad={p.requires_grad}")

    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    trainer = LLaVATrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)
    trainer.train()
    trainer.save_state()
    print(f"AFTER TRAIN: trainer config do_sample={trainer.model.config.do_sample}, temperature={trainer.model.config.temperature}")

    model.config.use_cache = True
    safe_save_model_for_hf_trainer(trainer=trainer,
                                   output_dir=training_args.output_dir)
    print(f"SAVE MODEL RESULT TO: {training_args.output_dir}")


if __name__ == "__main__":
    test_save_pt()

