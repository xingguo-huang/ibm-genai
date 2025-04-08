from transformers import AutoProcessor, AutoModelForVision2Seq
from huggingface_hub import hf_hub_download
import torch
from PIL import Image

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and processor
model_path = "ibm-granite/granite-vision-3.2-2b"
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForVision2Seq.from_pretrained(model_path).to(device)

# Load example image from Hugging Face
img_path = hf_hub_download(repo_id=model_path, filename='example.png')
image = Image.open(img_path).convert("RGB")

# Define conversation prompt
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": img_path},
            {"type": "text", "text": "What is the highest scoring model on ChartQA and what is its score?"},
        ],
    },
]

# Tokenize and process
inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_tensors="pt",
    return_dict=True
).to(device)

# Generate response
output = model.generate(**inputs, max_new_tokens=100)
print(processor.decode(output[0], skip_special_tokens=True))