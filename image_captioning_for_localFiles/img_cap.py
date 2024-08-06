import os 
import glob
import requests
from PIL import Image 
from transformers import AutoProcessor, BlipForConditionalGeneration


# Load the pretrained processor and model
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Specify the directory where your images are
image_dir = "./images"
image_exts = ["jpg", "jpeg", "png"]  # specify the image file extensions to search for

# Open a file to write the captions
with open("captions.txt", "w") as caption_file:
# Iterate over each image file in the directory
    for image_ext in image_exts:
        for img_path in glob.glob(os.path.join(image_dir, f"*.{image_ext}")):
            # Load your image
            raw_image = Image.open(img_path).convert('RGB')

            inputs = processor(raw_image , return_tensors="pt")
            outputs = model.generate(**inputs , max_new_tokens=50)
            caption = processor.decode(outputs[0] , skip_special_tokens=True)

            caption_file.write(f"{os.path.basename(img_path)} : {caption}\n")