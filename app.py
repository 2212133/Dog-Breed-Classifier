import streamlit as st
from PIL import Image, UnidentifiedImageError
import torch
import torch.nn as nn
from torchvision import models, transforms

@st.cache_resource
def load_model_and_mapping(ckpt_path="resnet_custom_head.pth"):
    # 1) Build the EXACT same architecture as training
    model = models.resnet101(weights=None)
    for p in model.parameters():
        p.requires_grad = False
    
    # Match the EXACT architecture from training (no extra dropout)
    model.fc = nn.Sequential(
        nn.Linear(2048, 256),        # From ResNet's output to 256 units
        nn.ReLU(),                   # Activation function
        nn.BatchNorm1d(256),         # Normalization for stability
        nn.Linear(256, 256),         # Additional dense layer
        nn.ReLU(),                   # Activation again
        nn.Dropout(0.3),             # Dropout to reduce overfitting
        nn.BatchNorm1d(256),         # Another batch norm layer
        nn.Linear(256, 120)          # Final layer: 120 output classes (NO extra dropout here)
    )

    # 2) Load checkpoint
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if "model_state_dict" not in ckpt or "class_to_idx" not in ckpt:
            st.error("Checkpoint is missing weights or class mapping.")
            st.stop()

        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        model.eval()

        # 3) Invert the mapping
        class_to_idx = ckpt["class_to_idx"]
        idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}

        return model, idx_to_class
    
    except FileNotFoundError:
        st.error(f"Model file '{ckpt_path}' not found. Please ensure the file is in the correct location.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

def tta_predict(model, pil_image, tta_transforms):
    """Apply each TTA transform to PIL image, run through model, accumulate logits."""
    device = torch.device("cpu")
    logits_sum = None
    
    with torch.no_grad():
        for transform in tta_transforms:
            # Apply transform to PIL image and add batch dimension
            img_tensor = transform(pil_image).unsqueeze(0)  # Shape: [1, 3, 224, 224]
            
            # Get model output
            logits = model(img_tensor.to(device))
            
            # Accumulate logits
            if logits_sum is None:
                logits_sum = logits
            else:
                logits_sum += logits
    
    # Average the logits
    return logits_sum / len(tta_transforms)

def main():
    st.title("🐶 Dog Breed Classifier with TTA")
    st.write("Upload an image of a dog to classify its breed!")

    # Load model once
    model, idx_to_class = load_model_and_mapping()

    # Define TTA transforms that match your TRAINING preprocessing
    # CRITICAL: NO normalization since training data wasn't normalized!
    tta_transforms = [
        # Original image - matches your validation transform
        transforms.Compose([
            transforms.Resize((224, 224)),  # Match your training: Resize to 224x224 directly
            transforms.ToTensor(),          # Convert to tensor [0, 1] - NO NORMALIZATION
        ]),
        # Horizontally flipped image
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=1.0),  # Always flip
            transforms.ToTensor(),                    # NO NORMALIZATION
        ]),
        # Rotated image (similar to training augmentation)
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(degrees=10),    # Match your training rotation
            transforms.ToTensor(),                    # NO NORMALIZATION
        ]),
        # Color jittered image (match training augmentation)
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),                    # NO NORMALIZATION
        ])
    ]

    uploaded = st.file_uploader("Upload a dog image (jpg/png)", type=["jpg", "jpeg", "png"])
    
    if uploaded is None:
        st.info("Please upload an image to get started!")
        return

    try:
        # Load and convert image to RGB
        pil_image = Image.open(uploaded).convert("RGB")
    except UnidentifiedImageError:
        st.error("Could not read the uploaded image. Please try a different file.")
        return
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        return

    # Display the uploaded image
    st.image(pil_image, caption="Uploaded Image", use_container_width=True)

    # Show a spinner while processing
    with st.spinner("Analyzing image..."):
        try:
            # Get predictions using Test Time Augmentation
            avg_logits = tta_predict(model, pil_image, tta_transforms)
            
            # Convert to probabilities
            probabilities = torch.softmax(avg_logits, dim=1).squeeze(0)
            
            # Get top-5 predictions
            top_k = 5
            top_probs, top_indices = torch.topk(probabilities, k=min(top_k, len(probabilities)))
            
            st.subheader("🏆 Top Predictions:")
            
            # Display predictions with confidence bars
            for rank, (prob, idx) in enumerate(zip(top_probs, top_indices), 1):
                breed_name = idx_to_class[idx.item()]
                confidence = prob.item() * 100
                
                # Format breed name (replace underscores with spaces and title case)
                formatted_breed = breed_name.replace('_', ' ').title()
                
                # Create columns for better layout
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(f"**{rank}. {formatted_breed}**")
                    st.progress(confidence / 100)
                
                with col2:
                    st.metric("Confidence", f"{confidence:.2f}%")
                
                st.write("---")
                
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.error("Please check that your model file 'dog_breed_model_with_names.pth' is in the correct location.")

    

if __name__ == "__main__":
    main()