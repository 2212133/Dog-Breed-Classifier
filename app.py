import streamlit as st
from PIL import Image, UnidentifiedImageError

def load_model():
    # Local imports only
    import torch
    import torch.nn as nn
    from torchvision import models

    # 1) Backbone without pretrained weights
    model = models.resnet101(weights=None)

    # 2) Freeze backbone
    for p in model.parameters():
        p.requires_grad = False

    # 3) Custom head as trained
    model.fc = nn.Sequential(
        nn.Linear(2048, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.3),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.3),
        nn.Linear(256, 5),
    )

    # 4) Load checkpoint (allow mismatches)
    ckpt = torch.load("resnet_custom_head.pth", map_location="cpu")
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    return model

def main():
    import torch
    from torchvision import transforms

    st.title("🐶 Dog Breed Classifier")

    # Load model once
        # Load model once
    model = load_model()
    class_names = ['scottish_deerhound', 'maltese_dog', 'afghan_hound', 'entlebucher',
 'bernese_mountain_dog', 'shih-tzu', 'great_pyrenees', 'pomeranian',
 'basenji', 'samoyed', 'tibetan_terrier', 'airedale', 'leonberg', 'cairn',
 'japanese_spaniel', 'beagle', 'australian_terrier', 'miniature_pinscher',
 'blenheim_spaniel', 'irish_wolfhound', 'saluki', 'lakeland_terrier',
 'papillon', 'norwegian_elkhound', 'whippet', 'siberian_husky', 'pug',
 'chow', 'italian_greyhound', 'pembroke', 'ibizan_hound', 'border_terrier',
 'newfoundland', 'lhasa', 'silky_terrier', 'dandie_dinmont',
 'bedlington_terrier', 'sealyham_terrier', 'rhodesian_ridgeback',
 'irish_setter', 'old_english_sheepdog', 'collie', 'boston_bull',
 'schipperke', 'kelpie', 'african_hunting_dog', 'bouvier_des_flandres',
 'english_foxhound', 'weimaraner', 'bloodhound', 'bluetick',
 'labrador_retriever', 'saint_bernard', 'chesapeake_bay_retriever',
 'norfolk_terrier', 'english_setter', 'greater_swiss_mountain_dog',
 'basset', 'irish_terrier', 'groenendael', 'kerry_blue_terrier',
 'yorkshire_terrier', 'scotch_terrier', 'wire-haired_fox_terrier',
 'gordon_setter', 'keeshond', 'west_highland_white_terrier', 'malamute',
 'mexican_hairless', 'toy_poodle', 'dingo', 'clumber', 'affenpinscher',
 'welsh_springer_spaniel', 'miniature_poodle', 'standard_poodle',
 'staffordshire_bullterrier', 'toy_terrier', 'miniature_schnauzer',
 'appenzeller', 'norwich_terrier', 'irish_water_spaniel', 'sussex_spaniel',
 'black-and-tan_coonhound', 'rottweiler', 'cardigan', 'shetland_sheepdog',
 'dhole', 'english_springer', 'german_short-haired_pointer',
 'bull_mastiff', 'borzoi', 'boxer', 'pekinese', 'great_dane',
 'cocker_spaniel', 'doberman', 'american_staffordshire_terrier',
 'malinois', 'brittany_spaniel', 'standard_schnauzer',
 'flat-coated_retriever', 'redbone', 'border_collie',
 'curly-coated_retriever', 'kuvasz', 'soft-coated_wheaten_terrier',
 'chihuahua', 'vizsla', 'french_bulldog', 'walker_hound',
 'german_shepherd', 'otterhound', 'giant_schnauzer', 'tibetan_mastiff',
 'golden_retriever', 'komondor', 'brabancon_griffon', 'eskimo_dog',
 'briard']




    # Preprocessing pipeline
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    uploaded = st.file_uploader(
        "Upload a dog image (JPEG or PNG)", 
        type=["jpg", "jpeg", "png"]
    )
    if not uploaded:
        return

    try:
        img = Image.open(uploaded).convert("RGB")
    except UnidentifiedImageError:
        st.error("Unsupported image format—please upload JPEG or PNG.")
        return

    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Inference
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        out = model(tensor)
        probs = torch.nn.functional.softmax(out, dim=1)
        conf, idx = torch.max(probs, 1)

    st.markdown(f"**Prediction:** {class_names[idx.item()]}")
    st.markdown(f"**Confidence:** {conf.item() * 100:.2f}%")

if __name__ == "__main__":
    main()
