import os

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import segmentation_models_pytorch as smp


MODEL_PATH = os.path.join(os.path.dirname(os.getcwd()), 'models')

class SegmentationModel(nn.Module):
    """
    A class representing a segmentation model for lung segmentation.

    This class encapsulates the functionality of a segmentation model based on the DeepLabV3Plus architecture.

    Attributes:
        model (nn.Module): The DeepLabV3Plus model for lung segmentation.
        preprocess (transforms.Compose): The preprocessing transformations applied to input images.

    Methods:
        initialize_model(): Initializes the DeepLabV3Plus model and loads the pre-trained weights.
        setup_preprocessing(): Sets up the preprocessing transformations.
        postprocess(output, original_size): Performs post-processing on the model output.
        forward(img): Performs forward pass through the model.

    """

    def __init__(self):
        super(SegmentationModel, self).__init__()
        self.initialize_model()
        self.setup_preprocessing()

    def initialize_model(self):
        """
        Initializes the DeepLabV3Plus model and loads the pre-trained weights.
        """
        ENCODER = 'resnet34'
        ENCODER_WEIGHTS = 'imagenet'
        NUM_CLASSES = 1
        ACTIVATION = 'sigmoid' 
        self.model = smp.DeepLabV3Plus(
            encoder_name=ENCODER, 
            encoder_weights=ENCODER_WEIGHTS,
            in_channels=1,
            classes=NUM_CLASSES, 
            activation=ACTIVATION,
        )

        state_dict = torch.load(os.path.join(MODEL_PATH, 'best_model_dlv3p.pth'), map_location=torch.device('cpu'))
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def setup_preprocessing(self):
        """
        Sets up the preprocessing transformations.
        """
        self.preprocess = transforms.Compose([transforms.ToTensor(),
                                 transforms.Resize((512, 512))])

    def postprocess(self, output, original_size):
        """
        Performs post-processing on the model output.

        Args:
            output (torch.Tensor): The model output tensor.
            original_size (tuple): The original size of the input image.

        Returns:
            output_image (PIL.Image.Image): The post-processed output image.

        """
        output = output.squeeze()
        output = torch.where(output > 0.5, torch.tensor(1.0), torch.tensor(0.0))
        output_image = Image.fromarray(output.byte().cpu().numpy())
        return output_image

    @torch.inference_mode()
    def forward(self, img):
        """
        Performs forward pass through the model.

        Args:
            img (PIL.Image.Image): The input image.

        Returns:
            output (torch.Tensor): The model output.

        """
        test_sample_image = self.preprocess(img)
        test_sample_image = test_sample_image[0:1, :, :]
        test_sample_image = test_sample_image.unsqueeze(0)
        test_sample_image = test_sample_image.to(next(self.model.parameters()).device)

        with torch.no_grad():
            output = self.model(test_sample_image)
        return output