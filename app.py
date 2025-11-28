# ==============================
# 🌌 DreamFrame Studio (Green + Black Edition)
# Turning your dreams into visual reality ✨
# ==============================

import torch
from diffusers import DiffusionPipeline
import gradio as gr

# Step 1: Load AI Model (Only once)
print("AI Model ko GPU par load kiya ja raha hai... Is mein kuch minute lag sakte hain.")
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
)
pipe.to("cuda")
print("Model tayar hai!")


# Step 2: Image generation function
def generate_image(prompt, negative_prompt, width, height):
    if not prompt:
        raise gr.Error("Please enter a prompt!")

    print(f"Image banai ja rahi hai: '{prompt}' ({width}x{height})")
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height
    ).images[0]

    print("Image ban gayi!")
    return image


# Step 3: Custom CSS (Green + Black Clean Theme)
custom_css = ""
/* ======== DREAMFRAME STUDIO - GREEN + BLACK THEME ======== */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

body {
  margin: 0;
  padding: 0;
  font-family: 'Poppins', sans-serif;
  background: #000000; /* Pure dark black */
  color: #ffffff;
  overflow-x: hidden;
}

/* Slight neon green edge glow for depth */
body::before {
  content: "";
  position: fixed;
  top: -50%;
  left: -50%;
  width: 200%;
  height: 200%;
  background: radial-gradient(circle at center, rgba(0,255,100,0.05), transparent 60%);
  z-index: -1;
}

/* Main container */
.gradio-container {
  background: rgba(15, 15, 15, 0.85) !important;
  backdrop-filter: blur(10px);
  border: 1px solid rgba(0, 255, 100, 0.15);
  border-radius: 20px !important;
  box-shadow: 0 0 40px rgba(0, 255, 100, 0.08);
  padding: 20px !important;
}

/* Title styling */
#app-title {
  font-size: 3em;
  font-weight: 700;
  text-align: center;
  background: linear-gradient(90deg, #00ff99, #00cc66);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  text-shadow: 0 0 25px rgba(0, 255, 100, 0.5);
}

/* Subtitle */
#app-subtitle {
  text-align: center;
  font-size: 1.2em;
  color: #b0b0b0;
  margin-top: -10px;
}

/* Inputs */
.gr-box {
  background-color: rgba(255, 255, 255, 0.05) !important;
  border: 1px solid rgba(0, 255, 100, 0.2) !important;
  border-radius: 10px !important;
}
.gr-input, .gr-slider {
  color: #ffffff !important;
  background: transparent !important;
}
.gr-label {
  color: #cccccc !important;
}

/* Buttons (same color as title) */
.gr-button {
  background: linear-gradient(90deg, #00ff99, #00cc66);
  border: none !important;
  border-radius: 40px !important;
  color: #000 !important;
  font-weight: 700 !important;
  box-shadow: 0 0 20px rgba(0, 255, 100, 0.5);
  transition: all 0.3s ease !important;
}
.gr-button:hover {
  transform: translateY(-2px);
  box-shadow: 0 0 40px rgba(0, 255, 100, 0.8);
}
#clear-btn {
  background: linear-gradient(90deg, #111, #222) !important;
  color: #fff !important;
}

/* Image output */
#image-output {
  border-radius: 15px !important;
  border: 2px solid rgba(0, 255, 100, 0.3) !important;
  box-shadow: 0 0 30px rgba(0, 255, 100, 0.2);
  transition: all 0.4s ease-in-out;
}
#image-output:hover {
  box-shadow: 0 0 45px rgba(0, 255, 100, 0.5);
}
""


# Step 4: Build Gradio Interface
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("<h1 id='app-title'>DreamFrame Studio</h1>", elem_id="app-title")
    gr.Markdown("<p id='app-subtitle'>Turning your dreams into visual reality ✨</p>", elem_id="app-subtitle")

    with gr.Row():
        with gr.Column(scale=2):
            prompt = gr.Textbox(label="Enter Your Prompt", placeholder="e.g., A futuristic city glowing green lights", lines=3)
            negative_prompt = gr.Textbox(label="Negative Prompt", placeholder="e.g., blurry, text, watermark")
            width = gr.Slider(minimum=512, maximum=1024, step=64, value=1024, label="Width")
            height = gr.Slider(minimum=512, maximum=1024, step=64, value=1024, label="Height")

            with gr.Row():
                clear_btn = gr.Button("Clear All", elem_id="clear-btn")
                generate_btn = gr.Button("✨ Generate Image")

        with gr.Column(scale=3):
            image_output = gr.Image(label="Your Masterpiece", elem_id="image-output")

    generate_btn.click(fn=generate_image, inputs=[prompt, negative_prompt, width, height], outputs=image_output)
    clear_btn.click(fn=lambda: ("", "", 1024, 1024, None),
                    inputs=None, outputs=[prompt, negative_prompt, width, height, image_output])

# Step 5: Launch the app
demo.launch(debug=True)
