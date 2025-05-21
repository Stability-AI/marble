import os
from typing import Dict

import numpy as np
import torch
from diffusers import ControlNetModel, StableDiffusionXLControlNetInpaintPipeline
from huggingface_hub import hf_hub_download, list_repo_files
from PIL import Image, ImageChops, ImageEnhance
from rembg import new_session, remove
from transformers import DPTForDepthEstimation, DPTImageProcessor

from ip_adapter_instantstyle import IPAdapterXL
from ip_adapter_instantstyle.utils import register_cross_attention_hook
from parametric_control_mlp import control_mlp

file_dir = os.path.dirname(os.path.abspath(__file__))
base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
image_encoder_path = "models/image_encoder"
ip_ckpt = "sdxl_models/ip-adapter_sdxl_vit-h.bin"
controlnet_path = "diffusers/controlnet-depth-sdxl-1.0"

# Cache for rembg sessions
_session_cache = None
CONTROL_MLPS = ["metallic", "roughness", "transparency", "glow"]


def get_session():
    global _session_cache
    if _session_cache is None:
        _session_cache = new_session()
    return _session_cache


def setup_control_mlps(
    features: int = 1024, device: str = "cuda", dtype: torch.dtype = torch.float16
) -> Dict[str, torch.nn.Module]:
    ret = {}
    for mlp in CONTROL_MLPS:
        ret[mlp] = setup_control_mlp(mlp, features, device, dtype)
    return ret


def setup_control_mlp(
    material_parameter: str,
    features: int = 1024,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
):
    net = control_mlp(features)
    net.load_state_dict(
        torch.load(os.path.join(file_dir, f"model_weights/{material_parameter}.pt"))
    )
    net.to(device, dtype=dtype)
    net.eval()
    return net


def download_ip_adapter():
    repo_id = "h94/IP-Adapter"
    target_folders = ["models/", "sdxl_models/"]
    local_dir = file_dir

    # Check if folders exist and contain files
    folders_exist = all(
        os.path.exists(os.path.join(local_dir, folder)) for folder in target_folders
    )

    if folders_exist:
        # Check if any of the target folders are empty
        folders_empty = any(
            len(os.listdir(os.path.join(local_dir, folder))) == 0
            for folder in target_folders
        )
        if not folders_empty:
            print("IP-Adapter files already downloaded. Skipping download.")
            return

    # List all files in the repo
    all_files = list_repo_files(repo_id)

    # Filter for files in the desired folders
    filtered_files = [
        f for f in all_files if any(f.startswith(folder) for folder in target_folders)
    ]

    # Download each file
    for file_path in filtered_files:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=file_path,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
        )
        print(f"Downloaded: {file_path} to {local_path}")


def setup_pipeline(
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
):
    download_ip_adapter()

    cur_block = ("up", 0, 1)

    controlnet = ControlNetModel.from_pretrained(
        controlnet_path, variant="fp16", use_safetensors=True, torch_dtype=dtype
    ).to(device)

    pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
        base_model_path,
        controlnet=controlnet,
        use_safetensors=True,
        torch_dtype=dtype,
        add_watermarker=False,
    ).to(device)

    pipe.unet = register_cross_attention_hook(pipe.unet)

    block_name = (
        cur_block[0]
        + "_blocks."
        + str(cur_block[1])
        + ".attentions."
        + str(cur_block[2])
    )

    print("Testing block {}".format(block_name))

    return IPAdapterXL(
        pipe,
        os.path.join(file_dir, image_encoder_path),
        os.path.join(file_dir, ip_ckpt),
        device,
        target_blocks=[block_name],
    )


def get_dpt_model(device: str = "cuda", dtype: torch.dtype = torch.float16):
    image_processor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
    model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas")
    model.to(device, dtype=dtype)
    model.eval()
    return model, image_processor


def run_dpt_depth(
    image: Image.Image, model, processor, device: str = "cuda"
) -> Image.Image:
    """Run DPT depth estimation on an image."""
    # Prepare image
    inputs = processor(images=image, return_tensors="pt").to(device, dtype=model.dtype)

    # Get depth prediction
    with torch.no_grad():
        depth_map = model(**inputs).predicted_depth

    # Now normalize to 0-1 range
    depth_map = (depth_map - depth_map.min()) / (
        depth_map.max() - depth_map.min() + 1e-7
    )
    depth_map = depth_map.clip(0, 1) * 255

    # Convert to PIL Image
    depth_map = depth_map.squeeze().cpu().numpy().astype(np.uint8)
    return Image.fromarray(depth_map).resize((1024, 1024))


def prepare_mask(image: Image.Image) -> Image.Image:
    """Prepare mask from image using rembg."""
    rm_bg = remove(image, session=get_session())
    target_mask = (
        rm_bg.convert("RGB")
        .point(lambda x: 0 if x < 1 else 255)
        .convert("L")
        .convert("RGB")
    )
    return target_mask.resize((1024, 1024))


def prepare_init_image(image: Image.Image, mask: Image.Image) -> Image.Image:
    """Prepare initial image for inpainting."""

    # Create grayscale version
    gray_image = image.convert("L").convert("RGB")
    gray_image = ImageEnhance.Brightness(gray_image).enhance(1.0)

    # Create mask inversions
    invert_mask = ImageChops.invert(mask)

    # Combine images
    grayscale_img = ImageChops.darker(gray_image, mask)
    img_black_mask = ImageChops.darker(image, invert_mask)
    init_img = ImageChops.lighter(img_black_mask, grayscale_img)

    return init_img.resize((1024, 1024))


def run_parametric_control(
    ip_model,
    target_image: Image.Image,
    edit_mlps: dict[torch.nn.Module, float],
    texture_image: Image.Image = None,
    num_inference_steps: int = 30,
    seed: int = 42,
    depth_map: Image.Image = None,
    mask: Image.Image = None,
) -> Image.Image:
    """Run parametric control with metallic and roughness adjustments."""
    # Get depth map
    if depth_map is None:
        model, processor = get_dpt_model()
        depth_map = run_dpt_depth(target_image, model, processor)
    else:
        depth_map = depth_map.resize((1024, 1024))

    # Prepare mask and init image
    if mask is None:
        mask = prepare_mask(target_image)
    else:
        mask = mask.resize((1024, 1024))

    if texture_image is None:
        texture_image = target_image

    init_img = prepare_init_image(target_image, mask)

    # Generate edit
    images = ip_model.generate_parametric_edits(
        texture_image,
        image=init_img,
        control_image=depth_map,
        mask_image=mask,
        controlnet_conditioning_scale=1.0,
        num_samples=1,
        num_inference_steps=num_inference_steps,
        seed=seed,
        edit_mlps=edit_mlps,
        strength=1.0,
    )

    return images[0]


def run_blend(
    ip_model,
    target_image: Image.Image,
    texture_image1: Image.Image,
    texture_image2: Image.Image,
    edit_strength: float = 0.0,
    num_inference_steps: int = 20,
    seed: int = 1,
    depth_map: Image.Image = None,
    mask: Image.Image = None,
) -> Image.Image:
    """Run blending between two texture images."""
    # Get depth map
    if depth_map is None:
        model, processor = get_dpt_model()
        depth_map = run_dpt_depth(target_image, model, processor)
    else:
        depth_map = depth_map.resize((1024, 1024))

    # Prepare mask and init image
    if mask is None:
        mask = prepare_mask(target_image)
    else:
        mask = mask.resize((1024, 1024))
    init_img = prepare_init_image(target_image, mask)

    # Generate edit
    images = ip_model.generate_edit(
        start_image=texture_image1,
        pil_image=texture_image1,
        pil_image2=texture_image2,
        image=init_img,
        control_image=depth_map,
        mask_image=mask,
        controlnet_conditioning_scale=1.0,
        num_samples=1,
        num_inference_steps=num_inference_steps,
        seed=seed,
        edit_strength=edit_strength,
        clip_strength=1.0,
        strength=1.0,
    )

    return images[0]
