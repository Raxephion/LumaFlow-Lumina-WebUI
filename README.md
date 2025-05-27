# LumaFlow-Lumina-WebUI - a highly optimised Lumina Image2.0 gradio web app for low VRAM systems (6GB+)

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) <!-- Assuming you'll keep the Apache 2.0 license -->

Welcome! This little app lets you harness the power of **Lumina Image 2.0** for your text-to-image creations, right on your own computer. It's designed to be super easy to get started, especially if you're not a fan of complicated setups.

Think of it as your local, friendly interface to a rather smart image model, designed to be (relatively) kind to your GPU.

## What's the Big Idea?

*   **Super Simple Setup:** Just download, click `install.bat`, then `launch.bat`. That's the dream!
*   **Lumina Image 2.0 Power:** Uses the impressive `Alpha-VLLM/Lumina-Image-2.0` model.
*   **Local & Private:** Your prompts and images stay on your machine.
*   **User-Friendly UI:** Clicky buttons and sliders, not scary code.
*   **VRAM Aware:** Includes optimizations like CPU offloading to help run on a wider range of GPUs.

## Features

*   Generate images from your text prompts.
*   Adjust image dimensions (Lumina likes multiples of 64, around 1024x1024 total pixels is a good start).
*   Control generation steps and guidance scale (CFG).
*   Use specific seeds for consistency or let randomness surprise you.
*   Images are saved automatically to an `output` folder.
*   Sleek, modern interface.

## Requirements (The Essentials)

### Hardware:
*   A **dedicated NVIDIA GPU** is strongly recommended (ideally with 6GB+ VRAM, though Lumina is a bit more forgiving than some giants).
*   *CPU Mode?* Possible....but don't bother.

### Software:
*   **Python** (version 3.9+ recommended). Get it from [python.org](https://www.python.org/downloads/) (tick "Add Python to PATH" on Windows).
*   **Git** (optional, for cloning). Get it from [git-scm.com](https://git-scm.com/downloads).

## The "Click-Click-Done" Installation (Windows)

This is the easy route for Windows users!

1.  **Download the App:**
    *   Go to the GitHub repository page: https://github.com/Raxephion/LumaFlow-Lumina-WebUI
    *   Click the green "<> Code" button, then "Download ZIP".
    *   Extract the downloaded ZIP file to a folder on your computer (e.g., `C:\LuminaApp`).

2.  **Run the Installer:**
    *   Navigate into the folder where you extracted the files.
    *   Double-click `install.bat`.
    *   A black window will appear and show progress. It will download Python libraries and set up a virtual environment. This might take a few minutes, especially the first time. Grab a beverage.
    *   Wait until it says "Installation complete" and prompts you to "Press any key to continue . . .".

3.  **Launch the App:**
    *   After the installation is done and you've pressed a key to close the installer window, double-click `launch.bat` in the same folder.
    *   The app will start. The first time, it will download the Lumina Image 2.0 model files (a few gigabytes). This is a one-time download per model.
    *   Once ready, it will show a URL like `http://127.0.0.1:7860`. Open this in your web browser.

That's it! You should be ready to generate images.

## Manual Installation (For Other OS or if `.bat` files aren't your jam)

If you're on macOS/Linux, or prefer doing things step-by-step:

1.  **Get the Code:**
    *   **With Git:**
        ```bash
        git clone https://github.com/Raxephion/LumaFlow-Lumina-WebUI.git
        cd LumaFlow-Lumina-WebUI
        ```
    *   **Download ZIP:** As described in Step 1 of the "Click-Click-Done" section, then navigate into the extracted folder with your terminal.

2.  **Create a Virtual Environment:**
    In your terminal, inside the project folder:
    ```bash
    python -m venv venv
    ```

3.  **Activate the Virtual Environment:**
    *   **Windows (in cmd or PowerShell):** `venv\Scripts\activate`
    *   **macOS/Linux (in bash/zsh):** `source venv/bin/activate`
    (Your terminal prompt should change to show `(venv)`)

4.  **Install Dependencies:**
    With the virtual environment active:
    ```bash
    pip install -r requirements.txt
    ```

5.  **Run the App:**
    ```bash
    python app.py
    ```
    Look for the `http://127.0.0.1:7860` URL in the terminal output and open it in your browser.

## About Lumina Image 2.0

This app uses `Alpha-VLLM/Lumina-Image-2.0`. It's part of the Lumina family of models, which are "next-generation foundation models for text-to-image generation, text-to-video generation, and multi-modal language understanding."
You can find more details on their [Hugging Face Hub page](https://huggingface.co/Alpha-VLLM/Lumina-Image-2.0).

## Troubleshooting

*   **"CUDA out of memory":** Try smaller image dimensions. Close other GPU-hungry apps.
*   **Slow first launch:** Probably downloading the model. Be patient.
*   **Errors during install/launch:**
    *   Ensure Python is installed correctly and added to PATH.
    *   Make sure you're running commands *inside the activated virtual environment* for manual setup.
    *   If `install.bat` fails, check its output for specific error messages.

## Disclaimer

This is a user-friendly wrapper for a powerful model. Use it responsibly. Image quality and performance can vary. Have fun experimenting!

## Thanks
Mad respect to:
🧠 QIN QI (ChinChyi) and stzhao (zhaoshitian) from Alpha-VLLM — your work is seriously inspiring and foundational to projects like this. You bring the sorcery, I bring the hype :)
