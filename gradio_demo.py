import gradio as gr
from PIL import Image

from marble import (
    get_session,
    run_blend,
    run_parametric_control,
    setup_control_mlps,
    setup_pipeline,
)

# Setup the pipeline and control MLPs
control_mlps = setup_control_mlps()
ip_adapter = setup_pipeline()
get_session()

# Load example images
EXAMPLE_IMAGES = {
    "blend": {
        "target": "input_images/context_image/beetle.png",
        "texture1": "input_images/texture/low_roughness.png",
        "texture2": "input_images/texture/high_roughness.png",
    },
    "parametric": {
        "target": "input_images/context_image/toy_car.png",
        "texture": "input_images/texture/metal_bowl.png",
    },
}


def blend_images(target_image, texture1, texture2, edit_strength):
    """Blend between two texture images"""
    result = run_blend(
        ip_adapter, target_image, texture1, texture2, edit_strength=edit_strength
    )
    return result


def parametric_control(
    target_image,
    texture_image,
    control_type,
    metallic_strength,
    roughness_strength,
    transparency_strength,
    glow_strength,
):
    """Apply parametric control based on selected control type"""
    edit_mlps = {}

    if control_type == "Roughness + Metallic":
        edit_mlps = {
            control_mlps["metallic"]: metallic_strength,
            control_mlps["roughness"]: roughness_strength,
        }
    elif control_type == "Transparency":
        edit_mlps = {
            control_mlps["transparency"]: transparency_strength,
        }
    elif control_type == "Glow":
        edit_mlps = {
            control_mlps["glow"]: glow_strength,
        }

    # Use target image as texture if no texture is provided
    texture_to_use = texture_image if texture_image is not None else target_image

    result = run_parametric_control(
        ip_adapter,
        target_image,
        edit_mlps,
        texture_to_use,
    )
    return result


# Create the Gradio interface
with gr.Blocks(
    title="MARBLE: Material Recomposition and Blending in CLIP-Space"
) as demo:
    gr.Markdown(
        """
        # MARBLE: Material Recomposition and Blending in CLIP-Space

        <div style="display: flex; justify-content: flex-start; gap: 10px;>
            <a href="https://arxiv.org/abs/"><img src="https://img.shields.io/badge/Arxiv-2501.04689-B31B1B.svg"></a>
            <a href="https://github.com/Stability-AI/marble"><img src="https://img.shields.io/badge/Github-Marble-B31B1B.svg"></a>
        </div>
        
        MARBLE is a tool for material recomposition and blending in CLIP-Space.
        We provide two modes of operation:
        - **Texture Blending**: Blend the material properties of two texture images and apply it to a target image.
        - **Parametric Control**: Apply parametric material control to a target image. You can either provide a texture image, transferring the material properties of the texture to the original image, or you can just provide a target image, and edit the material properties of the original image.
        """
    )

    with gr.Row(variant="panel"):
        with gr.Tabs():
            with gr.TabItem("Texture Blending"):
                with gr.Row(equal_height=False):
                    with gr.Column():
                        with gr.Row():
                            texture1 = gr.Image(label="Texture 1", type="pil")
                            texture2 = gr.Image(label="Texture 2", type="pil")
                        edit_strength = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.5,
                            step=0.1,
                            label="Blend Strength",
                        )
                    with gr.Column():
                        with gr.Row():
                            target_image = gr.Image(label="Target Image", type="pil")
                            blend_output = gr.Image(label="Blended Result")
                        blend_btn = gr.Button("Blend Textures")

                # Add examples for blending
                gr.Examples(
                    examples=[
                        [
                            Image.open(EXAMPLE_IMAGES["blend"]["target"]),
                            Image.open(EXAMPLE_IMAGES["blend"]["texture1"]),
                            Image.open(EXAMPLE_IMAGES["blend"]["texture2"]),
                            0.5,
                        ]
                    ],
                    inputs=[target_image, texture1, texture2, edit_strength],
                    outputs=blend_output,
                    fn=blend_images,
                    cache_examples=True,
                )

                blend_btn.click(
                    fn=blend_images,
                    inputs=[target_image, texture1, texture2, edit_strength],
                    outputs=blend_output,
                )

            with gr.TabItem("Parametric Control"):
                with gr.Row(equal_height=False):
                    with gr.Column():
                        with gr.Row():
                            target_image_pc = gr.Image(label="Target Image", type="pil")
                            texture_image_pc = gr.Image(
                                label="Texture Image (Optional - uses target image if not provided)",
                                type="pil",
                            )
                        control_type = gr.Dropdown(
                            choices=["Roughness + Metallic", "Transparency", "Glow"],
                            value="Roughness + Metallic",
                            label="Control Type",
                        )

                        metallic_strength = gr.Slider(
                            minimum=-20,
                            maximum=20,
                            value=0,
                            step=0.1,
                            label="Metallic Strength",
                            visible=True,
                        )
                        roughness_strength = gr.Slider(
                            minimum=-1,
                            maximum=1,
                            value=0,
                            step=0.1,
                            label="Roughness Strength",
                            visible=True,
                        )
                        transparency_strength = gr.Slider(
                            minimum=0,
                            maximum=4,
                            value=0,
                            step=0.1,
                            label="Transparency Strength",
                            visible=False,
                        )
                        glow_strength = gr.Slider(
                            minimum=0,
                            maximum=3,
                            value=0,
                            step=0.1,
                            label="Glow Strength",
                            visible=False,
                        )
                        control_btn = gr.Button("Apply Control")

                    with gr.Column():
                        control_output = gr.Image(label="Result")

                def update_slider_visibility(control_type):
                    return [
                        gr.update(visible=control_type == "Roughness + Metallic"),
                        gr.update(visible=control_type == "Roughness + Metallic"),
                        gr.update(visible=control_type == "Transparency"),
                        gr.update(visible=control_type == "Glow"),
                    ]

                control_type.change(
                    fn=update_slider_visibility,
                    inputs=[control_type],
                    outputs=[
                        metallic_strength,
                        roughness_strength,
                        transparency_strength,
                        glow_strength,
                    ],
                    show_progress=False,
                )

                # Add examples for parametric control
                gr.Examples(
                    examples=[
                        [
                            Image.open(EXAMPLE_IMAGES["parametric"]["target"]),
                            Image.open(EXAMPLE_IMAGES["parametric"]["texture"]),
                            "Roughness + Metallic",
                            0,  # metallic_strength
                            0,  # roughness_strength
                            0,  # transparency_strength
                            0,  # glow_strength
                        ],
                        [
                            Image.open(EXAMPLE_IMAGES["parametric"]["target"]),
                            Image.open(EXAMPLE_IMAGES["parametric"]["texture"]),
                            "Roughness + Metallic",
                            20,  # metallic_strength
                            0,  # roughness_strength
                            0,  # transparency_strength
                            0,  # glow_strength
                        ],
                        [
                            Image.open(EXAMPLE_IMAGES["parametric"]["target"]),
                            Image.open(EXAMPLE_IMAGES["parametric"]["texture"]),
                            "Roughness + Metallic",
                            0,  # metallic_strength
                            1,  # roughness_strength
                            0,  # transparency_strength
                            0,  # glow_strength
                        ],
                    ],
                    inputs=[
                        target_image_pc,
                        texture_image_pc,
                        control_type,
                        metallic_strength,
                        roughness_strength,
                        transparency_strength,
                        glow_strength,
                    ],
                    outputs=control_output,
                    fn=parametric_control,
                    cache_examples=True,
                )

                control_btn.click(
                    fn=parametric_control,
                    inputs=[
                        target_image_pc,
                        texture_image_pc,
                        control_type,
                        metallic_strength,
                        roughness_strength,
                        transparency_strength,
                        glow_strength,
                    ],
                    outputs=control_output,
                )

if __name__ == "__main__":
    demo.launch()
