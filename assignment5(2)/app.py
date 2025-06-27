# app.py

import torch
import gradio as gr
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import os

# --- 1. Model Loading ---
# We select a device to run the model on (GPU if available, otherwise CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the pretrained BLIP model and its processor from Hugging Face.
# This model is specifically designed for image-to-text generation.
# Note: This model is several gigabytes and will be downloaded on first run.
print("Loading BLIP model, this may take a while...")
model_name = "Salesforce/blip-image-captioning-large"
processor = BlipProcessor.from_pretrained(model_name)
# We load the model in float16 to save memory and speed up inference.
model = BlipForConditionalGeneration.from_pretrained(
    model_name, torch_dtype=torch.float16
).to(device)
print("BLIP model loaded successfully.")

# --- 2. The Core Generation Function ---
def generate_caption(image: Image.Image, use_beam_search: bool):
    """
    This function takes a PIL image and generates a caption using the BLIP model.
    It allows toggling between greedy search and beam search.
    """
    # Handle the case where the user doesn't upload an image
    if image is None:
        return "Please upload an image first."

    # The processor prepares the image for the model.
    # No text prompt is needed for unconditional image captioning.
    inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)

    # Generate captions using the model
    if use_beam_search:
        # Beam search generates multiple sequences and picks the most likely one.
        # It's slower but often produces more coherent and descriptive captions.
        generated_ids = model.generate(
            **inputs,
            max_length=50,
            num_beams=5,
            early_stopping=True
        )
    else:
        # Greedy search picks the most likely next word at each step.
        # It's faster but can sometimes lead to repetitive or less natural captions.
        generated_ids = model.generate(**inputs, max_length=50)

    # Decode the generated IDs back into a text string.
    # skip_special_tokens=True removes tokens like [CLS] and [SEP].
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    
    return caption

# --- 3. Building the Gradio Interface ---

# Define the input and output components for the web UI
image_input = gr.Image(type="pil", label="Upload an Image")
beam_search_toggle = gr.Checkbox(label="Use Beam Search (Higher Quality, Slower)", value=True)
output_textbox = gr.Textbox(label="Generated Caption", lines=2)

# Get list of example image paths if the 'examples' folder exists
example_paths = []
if os.path.exists("examples"):
    example_paths = [os.path.join("examples", fname) for fname in os.listdir("examples")]

# Create the Gradio interface
iface = gr.Interface(
    fn=generate_caption,
    inputs=[image_input, beam_search_toggle],
    outputs=output_textbox,
    title="Advanced Image Captioning with BLIP",
    description=(
        "This is a true generative image captioning model (Salesforce BLIP). "
        "Upload an image, and the model will generate a descriptive caption from scratch. "
        "The underlying architecture uses a Vision Transformer (ViT) encoder and a text decoder, "
        "fused with cross-attention for high-quality results."
    ),
    examples=[[path, True] for path in example_paths] if example_paths else None,
    allow_flagging="never"
)

# --- 4. Launching the App ---
if __name__ == "__main__":
    iface.launch()