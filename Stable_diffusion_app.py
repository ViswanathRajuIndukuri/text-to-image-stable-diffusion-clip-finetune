# app.py
import os
import torch
import streamlit as st
from io import BytesIO
import base64

from transformers import (
    CLIPProcessor,
    CLIPTextModel,
    CLIPImageProcessor,
)
from diffusers import (
    StableDiffusionPipeline,
    AutoencoderKL,
    UNet2DConditionModel,
    LMSDiscreteScheduler,
)
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# — Page config & wide layout —
st.set_page_config(
    page_title="Stable Diffusion",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_resource
def load_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    proc = CLIPProcessor.from_pretrained(
        "openai/clip-vit-large-patch14", clean_up_tokenization_spaces=True
    )
    vae = AutoencoderKL.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="vae"
    )
    unet = UNet2DConditionModel.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="unet"
    )
    text_enc = CLIPTextModel.from_pretrained("../fine_tuned_clip_40k")
    scheduler = LMSDiscreteScheduler.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="scheduler"
    )
    safety = StableDiffusionSafetyChecker.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="safety_checker"
    )
    img_proc = CLIPImageProcessor.from_pretrained(
        "openai/clip-vit-base-patch32"
    )

    pipe = StableDiffusionPipeline(
        vae=vae,
        unet=unet,
        text_encoder=text_enc,
        tokenizer=proc.tokenizer,
        scheduler=scheduler,
        safety_checker=safety,
        feature_extractor=img_proc,
    ).to(device)

    return pipe

def main():
    st.title("🖼️ Text-to-Image Generation Using Stable Diffusion with Fine-Tuned CLIP")
    st.write("Generate images from text with customizable inference settings.")

    # — Sidebar Inputs —
    with st.sidebar:
        st.header("🔧 Settings")
        examples = [
            "A red light and street sign in front of a palm tree",
            "An astronaut riding a horse on Mars",
            "A serene lake at sunrise in watercolor style",
            "Cyberpunk city at night with neon lights",
        ]

        prompt = st.text_input(
            "Prompt",
            value="",
            placeholder="Type your prompt here...",
        )

        # Example selector
        sel = st.selectbox(
            "Or choose an example",
            ["-- none --"] + examples,
        )
        if sel != "-- none --":
            prompt = sel

        with st.expander("Advanced options"):
            num_steps = st.slider(
                "Inference steps",
                min_value=10,
                max_value=100,
                value=50,
                help="More steps → finer detail but slower",
            )
            guidance = st.slider(
                "Guidance scale",
                min_value=1.0,
                max_value=12.0,
                value=7.5,
                step=0.5,
                help="Higher → closer to prompt",
            )

        generate = st.button("▶️ Generate", use_container_width=True)

    # — Main Content —
    if generate:
        if not prompt:
            st.sidebar.error("🔴 Please enter or select a prompt.")
        else:
            with st.spinner("Rendering…"):
                pipe = load_pipeline()
                out = pipe(
                    prompt,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance,
                )
                img = out.images[0]

            # two columns: image + metadata/history
            col1, col2 = st.columns([3, 1])
            with col1:
                # Convert PIL image to base64
                buf = BytesIO()
                img.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode()

                # HTML + CSS for a responsive image
                img_html = f"""
                <div style="text-align:center">
                <img
                    src="data:image/png;base64,{b64}"
                    style="
                    max-width: 100%;
                    max-height: 80vh;
                    object-fit: contain;
                    "
                />
                </div>
                """

                st.markdown(img_html, unsafe_allow_html=True)
            with col2:
                st.subheader("⚙️ Prompt details")
                st.markdown(f"**Prompt:**\n>{prompt}")
                st.markdown(f"- **Steps:** {num_steps}\n- **Guidance:** {guidance:.1f}")

    else:
        st.info("Enter a prompt and click **Generate** to begin.")

if __name__ == "__main__":
    main()