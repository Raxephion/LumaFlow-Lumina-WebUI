import os
import uuid
import gradio as gr
from diffusers import LuminaNextVQPipeline # MODIFIED: Changed pipeline
import torch
from PIL import Image
from pathvalidate import sanitize_filename
import gc # Garbage Collector

# --- Configuration ---
# Ensure output and model directories exist
os.makedirs("output", exist_ok=True)
os.makedirs("models", exist_ok=True)

MODEL_FOLDER = "models"
OUTPUT_FOLDER = "output"
MODEL_NAME = "Alpha-VLLM/Lumina-Image-2.0" # MODIFIED: Changed model name

# Determine device and dtype
if torch.cuda.is_available():
    try:
        torch.cuda.set_device(0) # Try primary GPU
        DEVICE = "cuda:0"
    except RuntimeError:
        print("CUDA device 0 not available or already in use. Trying cuda:1 if available.")
        try:
            torch.cuda.set_device(1)
            DEVICE = "cuda:1"
        except RuntimeError:
            print("CUDA device 1 not available. Falling back to CPU.")
            DEVICE = "cpu"
else:
    DEVICE = "cpu"

# MODIFIED: Using torch.bfloat16 if CUDA and Ampere+ might be slightly better,
# but float16 is a good default for wider CUDA compatibility.
TORCH_DTYPE = torch.float16 if DEVICE.startswith("cuda") else torch.float32
# VARIANT = "fp16" if DEVICE.startswith("cuda") else None # MODIFIED: Removed VARIANT as Lumina examples don't use it

# --- Global Variables ---
pipeline = None
last_seed = -1 # Initialize with -1 for random

# --- Model Loading ---
def load_lumina_pipeline(): # MODIFIED: Renamed function
    global pipeline
    if pipeline is None:
        print(f"Loading Lumina model: {MODEL_NAME} to {DEVICE} with {TORCH_DTYPE}...")
        try:
            # MODIFIED: Removed VARIANT from pipeline_args
            pipeline_args = {"cache_dir": MODEL_FOLDER, "torch_dtype": TORCH_DTYPE}

            # MODIFIED: Changed to LuminaNextVQPipeline
            pipeline = LuminaNextVQPipeline.from_pretrained(MODEL_NAME, **pipeline_args)

            if DEVICE.startswith("cuda"):
                print("Enabling model CPU offload for lower VRAM usage...")
                pipeline.enable_model_cpu_offload() # Key for low VRAM
            else:
                pipeline.to(DEVICE) # Move to CPU if not using CUDA offload

            print("Lumina model loaded successfully.") # MODIFIED: Text change
        except Exception as e:
            print(f"Error loading Lumina model: {e}") # MODIFIED: Text change
            pipeline = None # Ensure pipeline is None if loading failed
            raise # Re-raise the exception to notify the user via Gradio
    return pipeline

# --- Core Generation Logic ---
def generate_image(prompt: str, width: int, height: int, num_inference_steps: int, seed: int, guidance_scale: float, progress=gr.Progress(track_tqdm=True)):
    global last_seed, pipeline

    if pipeline is None:
        gr.Error("Model is not loaded. Please check console logs and restart the app.")
        return None, "Error: Model not loaded.", -1, "Error"

    current_seed = int(seed)
    if current_seed == -1:
        current_seed = torch.randint(0, 2**32 - 1, (1,)).item()
    last_seed = current_seed

    # For Lumina, generator on CPU is fine, or can be moved to GPU if preferred and VRAM allows fully.
    # Sticking to CPU for generator for consistency with previous script and max reproducibility.
    generator = torch.Generator(device="cpu").manual_seed(current_seed)

    status_message = "Generating..."
    try:
        print(f"Generating image with seed: {current_seed}, W: {width}, H: {height}, Steps: {num_inference_steps}, CFG: {guidance_scale}")
        pil_image = pipeline(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            generator=generator,
            guidance_scale=guidance_scale,
        ).images[0]

        # MODIFIED: Default filename prefix
        safe_prompt_segment = sanitize_filename(prompt[:50] if prompt else "lumina_img")
        if not safe_prompt_segment.strip():
            safe_prompt_segment = "lumina_img"
        
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{safe_prompt_segment}_{current_seed}_{unique_id}.png"
        filepath = os.path.join(OUTPUT_FOLDER, filename)
        
        pil_image.save(filepath, 'PNG')
        status_message = f"Image saved as: {filepath}"
        print(status_message)

        if DEVICE.startswith("cuda"):
            torch.cuda.empty_cache()
        gc.collect()

        return pil_image, current_seed, status_message

    except Exception as e:
        print(f"Error during image generation: {e}")
        import traceback
        traceback.print_exc()
        status_message = f"Error: {str(e)}"
        return None, current_seed, status_message

# --- UI Helper Functions ---
def reset_seed_value():
    return -1

def reuse_last_seed_value():
    global last_seed
    return last_seed if last_seed is not None else -1

# --- Gradio Interface ---
try:
    load_lumina_pipeline() # MODIFIED: Call renamed function
except Exception as e:
    print(f"Failed to load model on startup: {e}. The app might not function correctly.")

theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.purple,
    secondary_hue=gr.themes.colors.orange,
    neutral_hue=gr.themes.colors.gray,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
)

with gr.Blocks(theme=theme, css="""
    .gradio-container { background-color: #f7f7f7; }
    .small-button { min-width: 0 !important; width: 3em; height: 3em; padding: 0.25em !important; line-height: 1; font-size: 1.2em; align-self: end; margin-left: 0.5em !important; }
    #seed_row .gr-form { display: flex; align-items: flex-end; }
    #seed_row .gr-number { flex-grow: 1; }
    .gr-group { border-radius: 12px !important; box-shadow: 0 4px 8px rgba(0,0,0,0.05) !important; background-color: white !important; padding: 20px !important; }
    h1 { font-size: 2.5em !important; color: *primary_600 !important; text-align: center; margin-bottom: 0.5em !important; }
    .gr-markdown p { font-size: 1.1em; color: *neutral_600; text-align: center; margin-bottom: 1.5em; }
""") as demo:

    # MODIFIED: UI Text
    gr.Markdown("# Lumina Image 2.0 ✨ Next-VQ Generation")
    gr.Markdown("Generate images with Lumina Image 2.0. Recommended resolution around 1024x1024 (or similar pixel count, WxH as multiples of 64).")

    with gr.Row():
        with gr.Column(scale=2, min_width=400):
            with gr.Group():
                gr.Markdown("### 🎨 Generation Settings")
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="e.g., A majestic lion surveying its kingdom, photorealistic",
                    lines=3
                )
                
                with gr.Row():
                    # MODIFIED: Default values for Lumina, though max 2048 might be pushing it.
                    # Lumina likes total pixels around 1M, and W/H multiples of 64.
                    width_slider = gr.Slider(label="Width (multiple of 64)", minimum=256, maximum=1536, value=1024, step=64)
                    height_slider = gr.Slider(label="Height (multiple of 64)", minimum=256, maximum=1536, value=1024, step=64)

                # MODIFIED: Defaults for Lumina
                steps_slider = gr.Slider(label="Inference Steps", minimum=10, maximum=100, value=50, step=1)
                guidance_slider = gr.Slider(label="Guidance Scale (CFG)", minimum=1.0, maximum=20.0, value=7.5, step=0.1)

                with gr.Row(elem_id="seed_row"):
                    seed_input = gr.Number(label="Seed (-1 for random)", value=-1, precision=0, interactive=True)
                    random_seed_button = gr.Button("🎲", elem_classes="small-button") # Tooltip removed earlier
                    reuse_seed_button = gr.Button("♻️", elem_classes="small-button")   # Tooltip removed earlier

                generate_button = gr.Button("Generate Image", variant="primary", scale=2)

        with gr.Column(scale=3, min_width=500):
            with gr.Group():
                gr.Markdown("### 🖼️ Generated Image")
                output_image = gr.Image(label="Output", type="pil", interactive=False, show_download_button=True, show_share_button=True)
                with gr.Accordion("Generation Details", open=False):
                    generated_seed_output = gr.Textbox(label="Used Seed", interactive=False)
                    status_output = gr.Textbox(label="Status / Filename", interactive=False, lines=2)

    generate_button.click(
        fn=generate_image,
        inputs=[prompt_input, width_slider, height_slider, steps_slider, seed_input, guidance_slider],
        outputs=[output_image, generated_seed_output, status_output],
        api_name="generate_image"
    )

    random_seed_button.click(fn=reset_seed_value, inputs=[], outputs=seed_input)
    reuse_seed_button.click(fn=reuse_last_seed_value, inputs=[], outputs=seed_input)

# --- Launch ---
if __name__ == "__main__":
    demo.queue(max_size=1, default_concurrency_limit=1) 
    demo.launch(debug=True, share=False)
