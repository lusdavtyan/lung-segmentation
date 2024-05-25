import io
from typing import Optional

import numpy as np
from PIL import Image
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from segmentation import SegmentationModel

app = FastAPI(
    title="Image Segmentation API",
    description="Processes uploaded images and returns segmented versions.",
    version="0.0.1",
    openapi_tags=[{"name": "Segment", "description": "API endpoints related to image segmentation"}]
)

@app.get("/", tags=["Greeting"])
def root():
    """Greet a user."""
    return {"message": "This is a page for lung segmentation."}


#############################################################################################################
########################################## SEGMENTATION #####################################################
#############################################################################################################


model = SegmentationModel()


@app.post("/segment-lung-image/", tags=["Segment"])
async def segment_lung_image(
    file: UploadFile = File(description="A required image file for lung segmentation.")
):
    """Receives an image file and segments the lung area using a predefined model, returning the segmented
    lung image as a combined PNG image (input image and segmented image side by side).

    Args:
    - **file** (UploadFile): The image file to segment. Must be in a valid image format (e.g., "image/png").

    Returns:
    - **StreamingResponse**: The combined image.
    """
    if not file.content_type.startswith("image/"):
        return JSONResponse(status_code=400, content={"message": "File provided is not an image."})

    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    image = image.resize((512, 512))

    mask = model(image)
    mask_np = mask.cpu().numpy()
    mask_np = (mask_np * 255).astype(np.uint8)
    mask_image = Image.fromarray(mask_np.squeeze())

    gap_width = 20
    combined_width = image.width + gap_width + mask_image.width
    combined_height = max(image.height, mask_image.height) + 40
    combined_image = Image.new("RGB", (combined_width, combined_height), "white")


    combined_image.paste(image, (0, 0))
    combined_image.paste(mask_image, (image.width + gap_width, 0))

    byte_arr = io.BytesIO()
    combined_image.save(byte_arr, format="PNG")
    byte_arr.seek(0)

    return StreamingResponse(byte_arr, media_type="image/png")


class ModelDetails(BaseModel):
    algorithm_name: str
    training_date: str
    dataset: Optional[list] = None
    research_papers: Optional[list] = None

@app.get("/model-details/", tags=["Model"])
def get_model_details():
    """Returns details about the model in use, including its algorithm name, related research papers,
    version number, training date, and the dataset used for training.

    Returns:
    - **ModelDetails**: A dictionary containing the model details.
    """
    model_details = ModelDetails(
        algorithm_name="DeepLabv3+",
        training_date="2024-05-21",
        dataset=["COVID-19 CT Lung and Infection Segmentation Dataset",
                 "Ma Jun, et al., April 2020. URL https://doi.org/10.5281/zenodo. 3757476."],
        research_papers=[
            "Paschalis Bizopoulos et al. Comprehensive comparison of deep learning models for lung and covid-19 lesion segmentation in ct scans. 09 2020.",
            "Liang-Chieh Chen, et al. Encoder-decoder with atrous separable convolution for semantic image segmentation. In Proceedings of the European conference on computer vision (ECCV), pages 801–818, 2018.",
            "Joao OB Diniz, et al. Segmentation and quantification of covid-19 using pulmonary vessels extraction and deep learning. Multimedia Tools and Applications, 80(19):29367–29399, 2021."
        ]
    )
    return model_details

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)