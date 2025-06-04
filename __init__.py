import sys
import os

sys.path.append(os.path.dirname(__file__))

import comfy.model_management
import torch
from PIL import Image
import numpy as np
from .marble import (
    setup_control_mlps,
    setup_pipeline,
    run_blend,
    run_parametric_control,
)


# Add conversion functions
def tensor_to_pil(tensor):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.squeeze(0)
        # Convert to numpy and scale to 0-255
        image = (tensor.cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(image)
    return tensor


def pil_to_tensor(pil_image):
    if isinstance(pil_image, Image.Image):
        # Convert PIL to numpy array
        image = np.array(pil_image)
        # Convert to tensor and normalize to 0-1
        tensor = torch.from_numpy(image).float() / 255.0
        tensor = tensor.unsqueeze(0)
        device = comfy.model_management.get_torch_device()
        tensor = tensor.to(device)
        return tensor
    return pil_image


MARBLE_CATEGORY = "marble"


class MarbleControlMLPLoader:
    CATEGORY = MARBLE_CATEGORY
    FUNCTION = "load"
    RETURN_NAMES = ["control_mlp"]
    RETURN_TYPES = ["CONTROL_MLP"]

    @classmethod
    def INPUT_TYPES(cls):
        return {}

    def load(self):
        device = comfy.model_management.get_torch_device()
        mlps = setup_control_mlps(device=device)
        return (mlps,)


class MarbleIPAdapterLoader:
    CATEGORY = MARBLE_CATEGORY
    FUNCTION = "load"
    RETURN_NAMES = ["ip_adapter"]
    RETURN_TYPES = ["IP_ADAPTER"]

    @classmethod
    def INPUT_TYPES(cls):
        return {}

    def load(self):
        device = comfy.model_management.get_torch_device()
        ip_adapter = setup_pipeline(device=device)
        return (ip_adapter,)


class MarbleBlendNode:
    CATEGORY = MARBLE_CATEGORY
    FUNCTION = "blend"
    RETURN_NAMES = ["image"]
    RETURN_TYPES = ["IMAGE"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ip_adapter": ("IP_ADAPTER",),
                "image": ("IMAGE",),
                "texture_image1": ("IMAGE",),
                "texture_image2": ("IMAGE",),
                "edit_strength": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "num_inference_steps": (
                    "INT",
                    {"default": 20, "min": 1, "max": 100, "step": 1},
                ),
                "seed": (
                    "INT",
                    {"default": 42, "min": 0, "max": 2147483647, "step": 1},
                ),
            },
            "optional": {
                "mask": ("MASK", {"default": None}),
                "depth_map": ("IMAGE", {"default": None}),
            },
        }

    def blend(
        self,
        ip_adapter,
        image,
        texture_image1,
        texture_image2,
        edit_strength,
        num_inference_steps,
        seed,
        mask=None,
        depth_map=None,
    ):
        # Convert all inputs to PIL
        pil_image = tensor_to_pil(image)
        pil_texture1 = tensor_to_pil(texture_image1)
        pil_texture2 = tensor_to_pil(texture_image2)
        pil_depth_map = tensor_to_pil(depth_map) if depth_map is not None else None

        result = run_blend(
            ip_adapter,
            pil_image,
            pil_texture1,
            pil_texture2,
            edit_strength=edit_strength,
            num_inference_steps=num_inference_steps,
            seed=seed,
            depth_map=pil_depth_map,
            mask=mask,
        )
        # Convert result back to tensor
        return (pil_to_tensor(result),)


class MarbleParametricControl:
    CATEGORY = MARBLE_CATEGORY
    FUNCTION = "parametric_control"
    RETURN_NAMES = ["image"]
    RETURN_TYPES = ["IMAGE"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ip_adapter": ("IP_ADAPTER",),
                "image": ("IMAGE",),
                "control_mlps": ("CONTROL_MLP",),
                "num_inference_steps": (
                    "INT",
                    {"default": 30, "min": 1, "max": 100, "step": 1},
                ),
                "seed": (
                    "INT",
                    {"default": 42, "min": 0, "max": 2147483647, "step": 1},
                ),
            },
            "optional": {
                "mask": ("MASK", {"default": None}),
                "texture_image": ("IMAGE", {"default": None}),
                "depth_map": ("IMAGE", {"default": None}),
                "metallic_strength": (
                    "FLOAT",
                    {"default": 0.0, "min": -20.0, "max": 20.0, "step": 0.1},
                ),
                "roughness_strength": (
                    "FLOAT",
                    {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05},
                ),
                "transparency_strength": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 4.0, "step": 0.1},
                ),
                "glow_strength": (
                    "FLOAT",
                    {"default": 0.0, "min": -1.0, "max": 3.0, "step": 0.1},
                ),
            },
        }

    def parametric_control(
        self,
        ip_adapter,
        image,
        control_mlps,
        num_inference_steps,
        seed,
        mask=None,
        texture_image=None,
        depth_map=None,
        metallic_strength=0.0,
        roughness_strength=0.0,
        transparency_strength=0.0,
        glow_strength=0.0,
    ):
        # Convert inputs to PIL
        pil_image = tensor_to_pil(image)
        pil_texture = (
            tensor_to_pil(texture_image) if texture_image is not None else None
        )
        pil_depth_map = tensor_to_pil(depth_map) if depth_map is not None else None

        edit_mlps = {}
        for mlp_name, strength in [
            ("metallic", metallic_strength),
            ("roughness", roughness_strength),
            ("transparency", transparency_strength),
            ("glow", glow_strength),
        ]:
            if mlp_name in control_mlps and strength != 0.0:
                edit_mlps[control_mlps[mlp_name]] = strength

        result = run_parametric_control(
            ip_adapter,
            pil_image,
            edit_mlps,
            texture_image=pil_texture,
            num_inference_steps=num_inference_steps,
            seed=seed,
            depth_map=pil_depth_map,
            mask=mask,
        )
        # Convert result back to tensor
        return (pil_to_tensor(result),)


NODE_CLASS_MAPPINGS = {
    "MarbleControlMLPLoader": MarbleControlMLPLoader,
    "MarbleIPAdapterLoader": MarbleIPAdapterLoader,
    "MarbleBlendNode": MarbleBlendNode,
    "MarbleParametricControl": MarbleParametricControl,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MarbleControlMLPLoader": "Marble Control MLP Loader",
    "MarbleIPAdapterLoader": "Marble IP Adapter Loader",
    "MarbleBlendNode": "Marble Blend Node",
    "MarbleParametricControl": "Marble Parametric Control",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
