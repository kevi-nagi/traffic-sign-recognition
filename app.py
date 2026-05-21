import streamlit as st

import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn

from PIL import Image

st.title(
    "Traffic Sign Recognition using Robust MobileNetV2"
)

st.write(
    "Upload a traffic sign image for prediction."
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

num_classes = 43

model = models.mobilenet_v2(weights=None)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    num_classes
)

model.load_state_dict(
    torch.load(
        'robust_mobilenet.pth',
        map_location=device
    )
)

model = model.to(device)

model.eval()


classes = {
    0: "Speed limit 20 km/h",
    1: "Speed limit 30 km/h",
    2: "Speed limit 50 km/h",
    3: "Speed limit 60 km/h",
    4: "Speed limit 70 km/h",
    5: "Speed limit 80 km/h",
    6: "End of speed limit 80 km/h",
    7: "Speed limit 100 km/h",
    8: "Speed limit 120 km/h",
    9: "No passing",
    10: "No passing for vehicles over 3.5 tons",
    11: "Right-of-way at intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5 tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve left",
    20: "Dangerous curve right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing for vehicles over 3.5 tons"
}

transform = transforms.Compose([

    transforms.Resize((64,64)),

    transforms.ToTensor()
])

uploaded_file = st.file_uploader(
    "Upload Traffic Sign Image",
    type=["jpg", "png", "jpeg"]
)


if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="Uploaded Traffic Sign",
        width=300
    )

    image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = torch.nn.functional.softmax(
            outputs,
            dim=1
        )

        confidence, predicted = torch.max(
            probabilities,
            1
        )

    prediction = classes[predicted.item()]

    confidence_score = confidence.item() * 100

    st.success(
        f"Predicted Traffic Sign: {prediction}"
    )

    st.info(
        f"Confidence: {confidence_score:.2f}%"
    )