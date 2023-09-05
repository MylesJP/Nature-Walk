### 1. Imports and class names setup ### 
import gradio as gr
import os
import torch
import timm
import json
from torch import nn
from torchvision import transforms
from model import create_effnetb2_model
from timeit import default_timer as timer
from typing import Tuple, Dict

# Setup class names
with open("class_names_numeric.txt", "r") as f: # reading them in from class_names.txt
    class_names = [flower_name.strip() for flower_name in  f.readlines()]
    
### 2. Model and transforms preparation ###    

# Create model
# effnetb2, effnetb2_transforms = create_effnetb2_model(
#     num_classes=102, # could also use len(class_names)
# )

# Load saved weights

effnetb2 = timm.create_model('efficientnet_b2', pretrained=True)
effnetb2.classifier = nn.Linear(1408, 102)

effnetb2_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

effnetb2.load_state_dict(
    torch.load(
        f="Flowers_102.pth",
        map_location=torch.device("cpu"),  # load to CPU
    )
)

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

### 3. Predict function ###

# Create predict function
def predict(img) -> Tuple[Dict, float]:
    """Transforms and performs a prediction on img and returns prediction and time taken.
    """
    # Start the timer
    start_time = timer()
    
    # Transform the target image and add a batch dimension
    img = effnetb2_transforms(img).unsqueeze(0)
    
    # Put model into evaluation mode and turn on inference mode
    effnetb2.eval()
    with torch.inference_mode():
        # Pass the transformed image through the model and turn the prediction logits into prediction probabilities
        pred_probs = torch.softmax(effnetb2(img), dim=1)
    
    # Create a prediction label and prediction probability dictionary for each prediction class (this is the required format for Gradio's output parameter)
    pred_labels_and_probs = {cat_to_name[class_names[i]]: float(pred_probs[0][i]) for i in range(len(class_names))}
    
    # Calculate the prediction time
    pred_time = round(timer() - start_time, 5)
    
    # Return the prediction dictionary and prediction time 
    return pred_labels_and_probs, pred_time

### 4. Gradio app ###

# Create title, description and article strings
title = "Nature Walk Flower Classifier"
description = "An EfficientNetB2 feature extractor computer vision model to classify images of flowers."
article = "Created by Myles Penner (https://github.com/MylesJP)"

# Create examples list from "examples/" directory
example_list = [["examples/" + example] for example in os.listdir("examples")]

# Create Gradio interface 
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Label(num_top_classes=5, label="Predictions"),
        gr.Number(label="Prediction time (s)"),
    ],
    examples=example_list,
    title=title,
    description=description,
    article=article,
)

# Launch to Gradio
demo.launch()
