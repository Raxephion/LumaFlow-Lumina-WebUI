import os
import uuid
import gradio as gr
from diffusers import Lumina2Pipeline # Using Lumina's specific pipeline (Corrected from LuminaNextVQPipeline based on previous fixes)
import torch
from PIL import Image
from pathvalidate import sanitize_filename
import gc # Garbage Collector

# --- Configuration ---
os.makedirs("output", exist_ok=True)
os.makedirs("models", exist_ok=True)

MODEL_FOLDER = "models"
OUTPUT_FOLDER = "output"
MODEL_NAME = "Alpha-VLLM/Lumina-Image-2.0" # Lumina model

# --- Device Configuration (GPU ONLY) ---
DEVICE = None
TORCH_DTYPE = None

if not torch.cuda.is_available():
    print("--------------------------------------------------------------------------")
    print("ERROR: NVIDIA CUDA is not available on this system.")
    print("This application requires an NVIDIA GPU with compatible CUDA drivers.")
    print("Lumina models cannot run on CPU. The application will not be functional.")
    print("Please check your NVIDIA driver installation and CUDA setup.")
    print("--------------------------------------------------------------------------")
    # Gradio interface might still load but model loading will fail and generation will be disabled.
else:
    # Attempt to set device, default to "cuda" (let PyTorch choose) if specific indices fail
    selected_device = "cuda" # Default, PyTorch will pick the default CUDA device
    try:
        torch.cuda.set_device(0) # Try primary GPU
        selected_device = "cuda:0"
        print("Using CUDA device 0.")
    except RuntimeError:
        print("CUDA device 0 not available or already in use. Trying cuda:1 if available.")
        try:
            torch.cuda.set_device(1)
            selected_device = "cuda:1"
            print("Using CUDA device 1.")
        except RuntimeError:
            print("CUDA device 1 also not available. Defaulting to 'cuda' (PyTorch will select one).")
            # selected_device remains "cuda"

    DEVICE = selected_device
    TORCH_DTYPE = torch.float16 # Standard for modern CUDA GPUs with these models
    print(f"CUDA is available. Using device: {DEVICE} with dtype: {TORCH_DTYPE}")

# --- Global Variables ---
pipeline = None
last_seed = -1 # Initialize with -1 for random

# --- Model Loading (GPU ONLY) ---
def load_lumina_pipeline():
    global pipeline
    if DEVICE is None or not DEVICE.startswith("cuda"):
        print("Model loading skipped: No CUDA-enabled GPU available or selected.")
        # pipeline remains None
        return None

    if pipeline is None:
        print(f"Loading Lumina model: {MODEL_NAME} to {DEVICE} with {TORCH_DTYPE}...")
        try:
            pipeline_args = {
                "cache_dir": MODEL_FOLDER,
                "torch_dtype": TORCH_DTYPE,
            }

            # Use the corrected pipeline name based on our previous debugging
            pipeline = Lumina2Pipeline.from_pretrained(MODEL_NAME, **pipeline_args)

            print("Enabling model CPU offload for potentially lower VRAM usage on CUDA device...")
            pipeline.enable_model_cpu_offload() # Offloads parts to CPU RAM, GPU for compute
            # For pure GPU (if enough VRAM): pipeline.to(DEVICE)

            print(f"Lumina model '{MODEL_NAME}' loaded successfully to {DEVICE}.")
        except Exception as e:
            print(f"ERROR: Could not load Lumina model '{MODEL_NAME}'. Exception: {e}")
            import traceback
            traceback.print_exc()
            pipeline = None # Ensure pipeline is None if loading failed
            # Re-raise to be caught by the startup sequence or to notify user via Gradio
            raise
    return pipeline

# --- Core Generation Logic ---
def generate_image(prompt: str, width: int, height: int, num_inference_steps: int, seed: int, guidance_scale: float, progress=gr.Progress(track_tqdm=True)):
    global last_seed, pipeline

    if pipeline is None:
        error_message = "Model is not loaded. "
        if DEVICE is None or not DEVICE.startswith("cuda"):
            error_message += "No CUDA GPU available or an error occurred during GPU detection. Please check console logs."
        else:
            error_message += "Model loading may have failed. Please check console logs and try restarting."
        gr.Error(error_message)
        return None, -1, error_message # Match expected number of outputs

    current_seed = int(seed)
    if current_seed == -1:
        current_seed = torch.randint(0, 2**32 - 1, (1,)).item()
    last_seed = current_seed

    # Generator on CPU is generally fine and common practice
    generator = torch.Generator(device="cpu").manual_seed(current_seed)

    status_message = "Generating..."
    try:
        print(f"Generating image with: Prompt='{prompt}', Seed={current_seed}, W={width}, H={height}, Steps={num_inference_steps}, CFG={guidance_scale}")

        pil_image = pipeline(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            generator=generator,
            guidance_scale=guidance_scale,
        ).images[0]

        safe_prompt_segment = sanitize_filename(prompt[:50] if prompt else "lumina_img")
        if not safe_prompt_segment.strip(): # Handle empty or whitespace-only prompts
            safe_prompt_segment = "lumina_img"

        unique_id = str(uuid.uuid4())[:8]
        filename = f"{safe_prompt_segment}_{current_seed}_{unique_id}.png"
        filepath = os.path.join(OUTPUT_FOLDER, filename)

        pil_image.save(filepath, 'PNG')
        status_message = f"Image saved: {filepath}"
        print(status_message)

        if DEVICE.startswith("cuda"):
            torch.cuda.empty_cache() # Clear VRAM
        gc.collect() # Python garbage collection

        return pil_image, current_seed, status_message

    except Exception as e:
        print(f"ERROR during image generation: {e}")
        import traceback
        traceback.print_exc()
        status_message = f"Error during generation: {str(e)}"
        # Return a placeholder or None for the image, and the error details
        return None, current_seed, status_message

# --- UI Helper Functions ---
def reset_seed_value():
    return -1

def reuse_last_seed_value():
    global last_seed
    return last_seed if last_seed != -1 else -1 # Ensure -1 is returned if last_seed is initial -1

# --- Gradio Interface ---
# Attempt to load the model on startup
model_load_error = None
if DEVICE and DEVICE.startswith("cuda"):
    try:
        print("Attempting to load Lumina model on startup...")
        load_lumina_pipeline()
        if pipeline is None:
            model_load_error = "Model could not be loaded. Check console for errors."
    except Exception as e:
        model_load_error = f"Failed to load model on startup: {e}. Check console."
        print(model_load_error)
elif DEVICE is None:
    model_load_error = "CUDA GPU not detected. Model cannot be loaded."
    print(model_load_error)

# Use the Gradio Soft theme
theme = gr.themes.Soft()

# Custom CSS block is removed.
# The `css` argument will be removed from `gr.Blocks()`

with gr.Blocks(theme=theme) as demo: # Removed css=css argument
    gr.Markdown("# LumaFLow - Lumina Image 2.0 Offline Image Generation (GPU Only)") # Updated title
    gr.Markdown("Generate images with Lumina (Alpha-VLLM/Lumina-Image-2.0). Requires CUDA GPU. W/H should be multiples of 64.")

    if model_load_error:
        gr.Error(f"Startup Error: {model_load_error}")

    with gr.Row():
        with gr.Column(scale=2, min_width=400):
            with gr.Group(): # gr.Group will now be styled by the Soft theme
                gr.Markdown("### 🎨 Generation Settings")
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="e.g., A majestic lion surveying its kingdom, photorealistic",
                    lines=3
                )

                with gr.Row():
                    # Lumina likes total pixels around 1M (e.g., 1024x1024). Max ~1.5k for width/height.
                    width_slider = gr.Slider(label="Width (multiple of 64)", minimum=256, maximum=1536, value=1024, step=64)
                    height_slider = gr.Slider(label="Height (multiple of 64)", minimum=256, maximum=1536, value=1024, step=64)

                # Lumina defaults from original script (good starting point)
                steps_slider = gr.Slider(label="Inference Steps", minimum=10, maximum=100, value=50, step=1)
                guidance_slider = gr.Slider(label="Guidance Scale (CFG)", minimum=1.0, maximum=20.0, value=7.5, step=0.1) # Lumina seems to work well with CFG 3-8

                with gr.Row(): # Removed elem_id="seed_row" as specific CSS targeting it is gone
                    seed_input = gr.Number(label="Seed (-1 for random)", value=-1, precision=0, interactive=True)
                    # Removed elem_classes and tooltip as they were part of the custom CSS/newer Gradio features
                    random_seed_button = gr.Button("🎲")
                    reuse_seed_button = gr.Button("♻️")


                generate_button = gr.Button("Generate Image", variant="primary", scale=2, interactive=(pipeline is not None))

        with gr.Column(scale=3, min_width=500):
            with gr.Group(): # gr.Group will now be styled by the Soft theme
                gr.Markdown("### 🖼️ Generated Image")
                output_image = gr.Image(label="Output", type="pil", interactive=False, show_download_button=True, show_share_button=False)
                with gr.Accordion("Generation Details", open=False):
                    generated_seed_output = gr.Textbox(label="Used Seed", interactive=False)
                    status_output = gr.Textbox(label="Status / Filename", interactive=False, lines=2)

    generate_button.click(
        fn=generate_image,
        inputs=[prompt_input, width_slider, height_slider, steps_slider, seed_input, guidance_slider],
        outputs=[output_image, generated_seed_output, status_output],
        api_name="generate_image"
    )

    random_seed_button.click(fn=reset_seed_value, inputs=None, outputs=seed_input)
    reuse_seed_button.click(fn=reuse_last_seed_value, inputs=None, outputs=seed_input)

# --- Launch ---
if __name__ == "__main__":
    if DEVICE is None or not DEVICE.startswith("cuda"):
        print("\nApplication will launch, but image generation will be disabled due to no CUDA GPU.")
        print("Please ensure you have a compatible NVIDIA GPU and drivers installed.\n")

    print("Launching Gradio Web UI...")
    demo.queue(max_size=10, default_concurrency_limit=1)
    demo.launch(debug=True, share=False)
