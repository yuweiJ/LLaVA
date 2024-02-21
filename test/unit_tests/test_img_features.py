from PIL import Image
import torch
import requests
from transformers import CLIPVisionModel, CLIPImageProcessor
from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower
from llava.mm_utils import get_anyres_image_grid_shape
#, get_anyres_batch_images
from llava.mm_utils import process_images
from llava.train.train import ModelArguments, DataArguments, init_img_grid_pinpoints
from llava.model.multimodal_encoder.builder import build_vision_tower
from llava.model.multimodal_projector.builder import build_vision_projector
from llava.model.llava_arch import unpad_image


def test_img_features():

    vision_tower_name = "openai/clip-vit-large-patch14-336"
    # model = CLIPVisionTower(vision_tower_name, args=None)
    
    model = CLIPVisionModel.from_pretrained(vision_tower_name)
    image_processor = CLIPImageProcessor.from_pretrained(vision_tower_name)
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # image = Image.open(requests.get(url, stream=True).raw).convert('RGB')
    # inputs = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

    image = Image.open(requests.get(url, stream=True).raw).convert('RGB')
    print(f"image size={image.size}, {type(image.size)}")

    img_grid_pinpoints = [[336, 672], [672, 336], [672, 672], [1008, 336], [336, 1008]]
    batch_images = get_anyres_batch_images(image, img_grid_pinpoints, 336)
    # inputs = [image_processor.preprocess(img, return_tensors='pt')['pixel_values'][0].unsqueeze(0) for img in batch_images]
    inputs = image_processor.preprocess(batch_images, return_tensors='pt')['pixel_values']
    print(f"YW_DEBUG: inputs type={type(inputs)},size={inputs.size()}")
    import sys
    sys.exit(0)
    
    concat_images = torch.cat([image for image in inputs], dim=0)
    print(f"YW_DEBUG: image size before preprocess: {type(concat_images)}, {concat_images.size()}")
    # inputs = image_processor(images=image, return_tensors="pt")['pixel_values'][0]

    vision_select_layer = -2
    img_features = []
    outputs = model(concat_images, output_hidden_states=True)
    print(f"YW_DEBUG: feature after vision tower: {type(outputs)}, {outputs.hidden_states}")
    select_feature = outputs.hidden_states[vision_select_layer]
    out_feature = select_feature[:, 1:]
    print(f"YW_DEBUG: feature after vision tower size={out_feature.size()}")


def test_vision_tower():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert('RGB')
    image_size = image.size
    print(f"image size={image.size}, {type(image.size)}")

    model_cfg = ModelArguments()
    model_cfg.vision_tower = "openai/clip-vit-large-patch14-336"
    model_cfg.mm_vision_select_layer = -2
    model_cfg.mm_projector_type = "mlp2x_gelu"
    model_cfg.mm_patch_merge_type = "spatial_unpad"
    data_args = DataArguments()
    data_args.image_aspect_ratio = 'anyres'
    data_args.image_grid_pinpoints_str = "[[336, 672], [672, 336], [672, 672], [1008, 336], [336, 1008]]"
    init_img_grid_pinpoints(data_args)

    vision_tower = build_vision_tower(model_cfg)
    vision_tower.load_model()
    model_cfg.mm_hidden_size = vision_tower.hidden_size
    model_cfg.hidden_size = 1024
    mm_projector = build_vision_projector(model_cfg)

    # model_name_or_path = "/mnt/intel/data/yuwei/vlm/LLaVA/ckpts/lmsys/vicuna-7b-v1.5"
    # model = LlavaLlamaForCausalLM.from_pretrained(model_name_or_path)
    # model.get_model().initialize_vision_modules(
    #     model_args=model_cfg
    # )
    # vision_tower = model.get_vision_tower()
    image_processor = vision_tower.image_processor
    image = process_images([image], image_processor, data_args)[0]
    print(f"test_vision_tower: processed_image size={image.size()}")
    images = [image]
    concat_images = torch.cat([image for image in images], dim=0)
    print(f"test_vision_tower: concat_images size={image.size()}")
    
    # image_features = self.encode_images(concat_images)
    image_feature = mm_projector(vision_tower(concat_images))
    print(f"test_vision_tower: image_features size={image_feature.size()}")

    if image_feature.shape[0] > 1:
        base_image_feature = image_feature[0]
        image_feature = image_feature[1:]
        # print(f"YW_DEBUG: spatial: base_image_feature: {base_image_feature.size()}, image_feature={image_feature.size()}")
        height = width = vision_tower.num_patches_per_side
        assert height * width == base_image_feature.shape[0]
        if data_args.image_aspect_ratio == 'anyres':
            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_size, data_args.image_grid_pinpoints, vision_tower.config.image_size)
            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
            print(f"YW_DEBUG: anyres: image_feature size={image_feature.size()}")
        else:
            raise NotImplementedError
        if 'unpad' in model_cfg.mm_patch_merge_type:
            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
            image_feature = unpad_image(image_feature, image_size)
            # image_feature = torch.cat((
            #     image_feature,
            #     self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
            # ), dim=-1)
            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
            print(f"YW_DEBUG: unpad: image_feature size={image_feature.size()}")
        else:
            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
            image_feature = image_feature.flatten(0, 3)
        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
        print(f"test_vision_tower: final image_features size={image_feature.size()}")


if __name__ == "__main__":
    # test_img_features()
    test_vision_tower()

