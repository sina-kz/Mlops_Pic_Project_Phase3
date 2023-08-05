# main.py
from fastapi import FastAPI
from fastapi import UploadFile , File
from torchvision.models import efficientnet_b0
from torchvision import transforms
import torch
import numpy as np
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware

from enum import Enum
from predictor import Predictor
# import pyuac

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class MappingType(Enum):
    MLP = 'mlp'
    Transformer = 'transformer'

model_weights_path = 'prefix_only_10_8_10-009.pt'
prefix_length_clip = 10
num_layers = 8
prefix_length = 10
prefix_size = 768
mapping_type = 'transformer'

mapping_type = {'mlp': MappingType.MLP, 'transformer': MappingType.Transformer}[mapping_type]
predictor = Predictor(model_weights_path, mapping_type=mapping_type,clip_length=prefix_length_clip, num_layers=num_layers, prefix_length=prefix_length, prefix_size=prefix_size)

# model = efficientnet_b0(pretrained=True)
# model.eval()

def model_predict(image, predictor):
    return predictor.predict(image, use_beam_search=False)

@app.post("/predict")
def predict(image: UploadFile = File(...)):
    image = Image.open(image.file)
    image = np.array(image.convert('RGB'))
    # reading the image With PIL.image
    # tensor = preprocess_image(image)
    output = model_predict(image, predictor)
    print(output)
    # _, predicted_idx = torch.max(output.data, 1)
    return {"class": output.replace("[CLS] ", "").replace("[SEP]", ".")}

# def preprocess_image(image):
#     # Preprocessing logic
#     tensor = transform(image).unsqueeze(0)
#     return tensor


if __name__ == "__main__":
    import uvicorn

    # if not pyuac.isUserAdmin():
    #     print("Re-launching as admin!")
    #     pyuac.runAsAdmin()

    uvicorn.run(app, host="0.0.0.0", port=8000)
