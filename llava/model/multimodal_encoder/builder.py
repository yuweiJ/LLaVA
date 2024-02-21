import os
from .clip_encoder import CLIPVisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    # vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    vision_tower_name = getattr(vision_tower_cfg, 'vision_tower_name', None)
    is_absolute_path_exists = os.path.exists(vision_tower_name)
    if is_absolute_path_exists or vision_tower_name.startswith("openai") or vision_tower_name.startswith("laion") or "ShareGPT4V" in vision_tower_name:
        return CLIPVisionTower(vision_tower_name, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower_name}')
