# app.py - Simplified 3-category waste classifier
import torch, timm
from torchvision import transforms
from PIL import Image
import gradio as gr
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# Load model
ckpt = torch.load("mobilenetv3_small.pt", map_location="cpu")
classes = ckpt["classes"]
model = timm.create_model(ckpt["model"], pretrained=False, num_classes=len(classes))
model.load_state_dict(ckpt["state_dict"]); model.eval()

# Map 30 fine-grained categories to 3 broad categories
CATEGORY_MAPPING = {
    "Recyclable ♻️": [
        "aerosol_cans", "aluminum_food_cans", "aluminum_soda_cans",
        "cardboard_boxes", "cardboard_packaging",
        "glass_beverage_bottles", "glass_cosmetic_containers", "glass_food_jars",
        "magazines", "newspaper", "office_paper",
        "plastic_detergent_bottles", "plastic_food_containers",
        "plastic_soda_bottles", "plastic_water_bottles",
        "steel_food_cans"
    ],
    "Organic 🌱": [
        "coffee_grounds", "eggshells", "food_waste", "tea_bags"
    ],
    "Trash 🗑️": [
        "clothing", "disposable_plastic_cutlery", "paper_cups",
        "plastic_cup_lids", "plastic_shopping_bags", "plastic_straws",
        "plastic_trash_bags", "shoes",
        "styrofoam_cups", "styrofoam_food_containers"
    ]
}

# Create reverse mapping
fine_to_broad = {}
for broad_cat, fine_cats in CATEGORY_MAPPING.items():
    for fine_cat in fine_cats:
        fine_to_broad[fine_cat] = broad_cat

val_tfms = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(), transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
])

def predict(img):
    if img is None:
        return {}
    
    # Get predictions for all 30 classes
    x = val_tfms(Image.fromarray(img)).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = logits.softmax(1).squeeze().tolist()
    
    # Aggregate probabilities into 3 broad categories
    broad_probs = {"Recyclable ♻️": 0.0, "Organic 🌱": 0.0, "Trash 🗑️": 0.0}
    for fine_cat, prob in zip(classes, probs):
        broad_cat = fine_to_broad.get(fine_cat, "Trash 🗑️")
        broad_probs[broad_cat] += prob
    
    # Also show top specific predictions for reference
    top_specific = sorted(list(zip(classes, probs)), key=lambda t: t[1], reverse=True)[:3]
    result_text = f"\n\n📋 Specific predictions:\n"
    for cat, prob in top_specific:
        result_text += f"  • {cat}: {prob*100:.1f}%\n"
    
    return broad_probs, result_text

# Create Gradio interface
with gr.Blocks(title="Waste Classifier - 3 Categories") as demo:
    gr.Markdown("# 🗑️ Waste Classification System")
    gr.Markdown("Upload an image to classify waste into: **Recyclable**, **Organic**, or **Trash**")
    
    with gr.Row():
        with gr.Column():
            img_input = gr.Image(type="numpy", label="Upload Waste Image")
            predict_btn = gr.Button("Classify", variant="primary")
        
        with gr.Column():
            label_output = gr.Label(label="Category Prediction", num_top_classes=3)
            details_output = gr.Textbox(label="Detailed Breakdown", lines=5)
    
    predict_btn.click(fn=predict, inputs=img_input, outputs=[label_output, details_output])
    img_input.change(fn=predict, inputs=img_input, outputs=[label_output, details_output])

demo.launch()
