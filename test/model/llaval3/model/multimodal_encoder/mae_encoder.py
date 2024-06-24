import torch
import torch.nn as nn
import sys
import timm
# import timm.layers.patch_embed

# sys.modules['timm.models.layers'] = timm.layers



class MAEVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        delay_load=False
        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            raise ValueError("What happened!")

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.vision_tower = timm.create_model('vit_base_patch16_384',pretrained=False)
        self.vision_tower.load_state_dict(torch.load('/data/wzp/wz/model/mae-vit-384-ep50-dict.pth'))
        self.vision_tower.requires_grad_(False)
        self.vision_tower.image_processor = None

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower.forward_features(image.to(device=self.device, dtype=self.dtype).unsqueeze(0))
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            # images = images.to(torch.bfloat16)
            image_forward_outs = self.vision_tower.forward_features(images.to(device=self.device))
            image_features = self.feature_select(image_forward_outs)#.to(images.dtype)

        return image_forward_outs

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return torch.float32

    @property
    def device(self):
        return torch.device('cuda')

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return 768

    @property
    def num_patches_per_side(self):
        return 24

    @property
    def num_patches(self):
        return 576



