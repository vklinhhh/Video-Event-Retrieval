from PIL import Image
import requests
from transformers import AutoProcessor, CLIPModel, AutoTokenizer
import os
import torch
import json
# Load the CLIP models and tokenizer
vision_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
#tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

# Define the path to the Keyframes directory
keyframes_dir = "/Users/mac/Downloads/Keyframes/L10"

# List all subdirectories (video-specific keyframes)
video_dirs = [d for d in os.listdir(keyframes_dir) if os.path.isdir(os.path.join(keyframes_dir, d))]

keyframe_features_dict = {}  # Dictionary to store keyframe features

for video_dir in video_dirs:
    keyframe_dir = os.path.join(keyframes_dir, video_dir)
    keyframe_files = [f for f in os.listdir(keyframe_dir) if f.endswith(".jpg") and not f.startswith("._")]
    
    for keyframe_file in keyframe_files:
        keyframe_path = os.path.join(keyframe_dir, keyframe_file)
        keyframe_image = Image.open(keyframe_path)
        keyframe_inputs = processor(images=keyframe_image, return_tensors="pt")
        keyframe_features = vision_model.get_image_features(**keyframe_inputs)
        # Modify the key to store without the full path
        relative_keyframe_path = os.path.join(video_dir, keyframe_file)
        
        # Store keyframe features in the dictionary
        keyframe_features_dict[relative_keyframe_path] = keyframe_features.tolist()

# Save keyframe features to a JSON file
with open("./data/keyframe_features_10.json", "w") as json_file:
    json.dump(keyframe_features_dict, json_file)