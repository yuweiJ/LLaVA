import os
import argparse
from llava.utils import disable_torch_init
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria
import torch


def simple_infer(args):
    disable_torch_init()
    # model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(args.model_path)
    print(f"model_name={model_name}")
    # tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path,
                                                                        #    args.model_base,
                                                                        #    model_name)
    # tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    # model = AutoModelForCausalLM.from_pretrained(args.model_path,
    #     torch_dtype=torch.float16).cuda()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    simple_infer(args)