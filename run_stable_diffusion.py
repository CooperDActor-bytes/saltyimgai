# run_stable_diffusion.py

import torch
from diffusers import StableDiffusionPipeline

# Load the model
model_id = "CompVis/stable-diffusion-v1-4-original"
pipe = StableDiffusionPipeline.from_pretrained(model_id)

# Generate an image based on a prompt
prompt = "A beautiful landscape with mountains"
image = pipe(prompt).images[0]

# Save the image
image.save("generated_image.png")
