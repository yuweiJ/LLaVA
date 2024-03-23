from PIL import Image
import torch
import requests
from transformers import CLIPVisionModel, CLIPImageProcessor
from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower
from llava.mm_utils import get_anyres_image_grid_shape


def get_anyres_batch_images(image, img_grid_pinpoints, vision_tower_size=336):
    batch_images = [image]  
    num_patches_width, num_patches_height = get_anyres_image_grid_shape(image.size, img_grid_pinpoints, vision_tower_size)
    img_w, img_h = image.size
    print(f"image size={img_w}, {img_h}")
    new_w = img_w // num_patches_width
    new_h = img_h // num_patches_height
    print(f"tile image size={new_w}, {new_h}")
    for j in range(num_patches_height):
        for i in range(num_patches_width):
            # width in: i*new_w:(i+1)*new_w, height in j*new_h:(j+1)*new_h
            # tile = image[j*new_h:(j+1)*new_h, i*new_w:(i+1)*new_w]
            tile = image.crop((i*new_w, j*new_h, (i+1)*new_w, (j+1)*new_h))
            batch_images.append(tile)
    return batch_images


def test_img_features():

    vision_tower_name = "openai/clip-vit-large-patch14-336"
    # model = CLIPVisionTower(vision_tower_name, args=None)
    
    model = CLIPVisionModel.from_pretrained(vision_tower_name)
    image_processor = CLIPImageProcessor.from_pretrained(vision_tower_name)
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # image = Image.open(requests.get(url, stream=True).raw).convert('RGB')
    # inputs = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

    image = Image.open(requests.get(url, stream=True).raw).convert('RGB')
    print(f"image size={image.size}")

    img_grid_pinpoints = [[336, 672], [672, 336], [672, 672], [1008, 336], [336, 1008]]
    batch_images = get_anyres_batch_images(image, img_grid_pinpoints, 336)
    inputs = [image_processor.preprocess(img, return_tensors='pt')['pixel_values'][0].unsqueeze(0) for img in batch_images]
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


if __name__ == "__main__":
    test_img_features()

