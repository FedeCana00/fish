
import torch
import numpy as np

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms
from PIL import Image


def my_gradcam(model, images, path):
    """
    We want to see in which areas of the images our model focuses most.
 
    Args:
        model   : our model.
        images  : the images to pass to the model.
        path    : where to save the generated images.
    """

    target_layers = [model.enc[-1]]
    input_tensor = images
    # Note: input_tensor can be a batch tensor with several images!

    # Construct the CAM object once, and then re-use it on many images:
    cam = GradCAM(model=model, target_layers=target_layers)

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)

    for i, _ in enumerate(images):
        # In this example grayscale_cam has ONLY ONE IMAGE in the batch:
        grayscale_cam_img = grayscale_cam[i, :]

        transformed_tensor = input_tensor[i]

        # Se il tensore contiene più immagini, selezionane solo una (es. la prima)
        #if input_tensor.dim() == 4:
        #    transformed_tensor = input_tensor[0]

        # Rimuovere una eventuale dimensione extra per il canale di colore
        #if transformed_tensor.dim() == 4:
        #    transformed_tensor = transformed_tensor.squeeze(0)

        # Rimuovere la normalizzazione
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        original_tensor = transformed_tensor.clone()
        for t, m, s in zip(original_tensor, mean, std):
            t.mul_(s).add_(m)

        # Converte il tensore in un'immagine PIL
        to_pil = transforms.ToPILImage()
        original_image = to_pil(original_tensor)

        # Ripristinare la dimensione originale
        resize_back = transforms.Resize((original_image.size[1], original_image.size[0]))
        original_image = np.array(resize_back(original_image)) / 255.0

        visualization = show_cam_on_image(original_image, grayscale_cam_img, use_rgb=True)

        # You can also get the model outputs without having to re-inference
        #model_outputs = cam.outputs

        img = Image.fromarray(visualization)
        img.save(path + f"/image{i}.png")