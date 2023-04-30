import logging
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import mlflow
import azureml.core
from azureml.core import Workspace, Dataset, Experiment, Run, Model

import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as T
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torchinfo import summary
from torchvision.utils import make_grid

import matplotlib.pyplot as plt
#%matplotlib inline

from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
from skimage.segmentation import mark_boundaries

import captum
from captum.attr import LayerGradCam, LayerAttribution
from captum.attr import visualization as viz
from captum.attr import Lime, LimeBase
from captum._utils.models.linear_model import SkLearnLinearRegression, SkLearnLasso
import torchvision.transforms as T
from captum.attr._core.lime import get_exp_kernel_similarity_function

import shutil
import json
import splitfolders
from textwrap import wrap
from datetime import datetime
from demoutils import train, plot_learning_curve, to_categorical, predict_loader, inverse_normalize, plot_confusion_matrix

from lime import lime_image
from lime import submodular_pick

#from skimage.segmentation import mark_boundariesimport
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as T
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torchinfo import summary
from torchvision.utils import make_grid
import os

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def concat_images(imga, imgb):
    """
    Combines two color image ndarrays side-by-side.
    """
    ha,wa = imga.shape[:2]
    hb,wb = imgb.shape[:2]
    max_height = np.max([ha, hb])
    total_width = wa+wb
    new_img = np.zeros(shape=(max_height, total_width, 3))
    new_img[:ha,:wa]=imga
    new_img[:hb,wa:wa+wb]=imgb
    return new_img

def concat_n_images(image_path_list):
    """
    Combines N color images from a list of image paths.
    """
    output = None
    for i, img_path in enumerate(image_path_list):
        img = plt.imread(img_path)[:,:,:3]
        if i==0:
            output = img
        else:
            output = concat_images(output, img)
    return output

def get_pil_transform(): 
    transf = T.Compose([
        T.Resize((256, 256)),
        T.CenterCrop(224)
    ])    

    return transf

def get_preprocess_transform():
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])     
    transf = T.Compose([
        T.ToTensor(),
        normalize
    ])    

    return transf    

def batch_predict(images):
    model.eval()
    batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch = batch.to(device)
    
    logits = model(batch)
    probs = F.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()

pill_transf = get_pil_transform()
preprocess_transform = get_preprocess_transform()

classCount = 4

model = models.squeezenet1_1(pretrained=False)
model.classifier[1] = nn.Conv2d(512, classCount, kernel_size=(1,1), stride=(1,1))
model.num_classes = classCount
    
model.load_state_dict(torch.load("brain-tumor-classification.pth", map_location=torch.device('cpu')))
model.eval()
summary = summary(model)
print("-----------------")
print(str(model.type))

explainer = lime_image.LimeImageExplainer()

exp_eucl_distance = get_exp_kernel_similarity_function('euclidean', kernel_width=1000)

lr_lime = Lime(
    model, 
    interpretable_model=SkLearnLinearRegression(),  # build-in wrapped sklearn Linear Regression
    similarity_func=exp_eucl_distance
)

img_height, img_width, channels = 224, 224, 3 # 256, 256, 3
rgb_means = (0.485, 0.456, 0.406) # (0.1781, 0.1781, 0.1781)
rgb_stds = (0.229, 0.224, 0.225) # (0.1936, 0.1936, 0.1936) 
batch_size = 32

transform_val = T.Compose([T.Resize((img_height, img_width)),
                           T.ToTensor(),   
                           T.Normalize(rgb_means, rgb_stds)    
                          ])

base_dir_tvt = "E:\PyTorch"
test_ds = ImageFolder(os.path.join(base_dir_tvt, 'explain-samples'), transform_val)
test_ldr = DataLoader(test_ds, batch_size)
print(test_ds.class_to_idx)

batch = iter(test_ldr)
images, labels = next(batch)
output = model(images)
probabilities = torch.nn.functional.softmax(output[0], dim=0)
#display_batch_of_images((images, labels), predictions)

y_true, y_pred, y_probs = predict_loader(model, test_ldr,  torch.device('cpu'))

i = 0
samples = {
'./explain-samples/glioma_tumor/glioma-1.jpg',
'./explain-samples/meningioma_tumor/meningioma-1.jpg',
'./explain-samples/meningioma_tumor/meningioma-2.jpg',
'./explain-samples/meningioma_tumor/meningioma-3.jpg',
'./explain-samples/no_tumor/no-tumor-1.jpg',
'./explain-samples/no_tumor/no-tumor-2.jpg',
'./explain-samples/no_tumor/no-tumor-3.jpg',
'./explain-samples/pituitary_tumor/pituitary-1.jpg',
'./explain-samples/pituitary_tumor/pituitary-2.jpg',
'./explain-samples/pituitary_tumor/pituitary-3.jpg'
}

for file in samples:
    print(file)
    i=i+1
    #Get test image for explanations
    test_img = Image.open(file)
    test_img_data = np.asarray(test_img)
    #plt.imshow(test_img_data)
    plt.axis('off')
    plt.show()

    model = model.to('cpu')
    model.eval()
    idx_to_labels = {idx : name for idx, name in enumerate(test_ds.classes)}

    # Preprocess for inference and explanations
    transform = T.Compose([T.Resize((img_height, img_width)), T.ToTensor()])
    transform_normalize = T.Normalize(rgb_means, rgb_stds)

    transformed_img = transform(test_img)
    input_img = transform_normalize(transformed_img)
    input_img = input_img.unsqueeze(0) 

    output = model(input_img)
    output = F.softmax(output, dim=1)
    print(output)
    prediction_score, pred_label_idx = torch.topk(output, 1)
    pred_label_idx.squeeze_()
    predicted_label=test_ldr.dataset.classes[pred_label_idx.item()]
    #predicted_label = idx_to_labels[pred_label_idx.item()]
    print('Predicted: {} ({:.4f})'.format(predicted_label, prediction_score.squeeze().item()))

    """ attributions_lgc = lr_lime.attribute(input_img, target=pred_label_idx,    n_samples=40,
        perturbations_per_eval=16,
        show_progress=True
    )

    print('Attribution range:', attributions_lgc.min().item(), 'to', attributions_lgc.max().item())
    upsamp_attr_lgc = LayerAttribution.interpolate(attributions_lgc, input_img.shape[2:])

    fig = viz.visualize_image_attr_multiple(upsamp_attr_lgc[0].cpu().permute(1,2,0).detach().numpy(),
                                        transformed_img.permute(1,2,0).numpy(),
                                        ["original_image", "blended_heat_map", "blended_heat_map", "masked_image"],
                                        ["all","positive","negative", "positive"],
                                        show_colorbar=True,
                                        titles=["Original", "Positive Attribution", "Negative Attribution", "Masked"],
                                        fig_size=(18, 6))

    fig[0].savefig("attrib.jpg")

    try:
        mlflow.log_figure(fig[0], 'LIME-explanations.png')
    except Exception as e: print(e) """

    #**********************

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(np.array(pill_transf(test_img)), 
                                            batch_predict, # classification function
                                            top_labels=1, 
                                            hide_color=0, 
                                            num_samples=1000) # number of images that will be sent to classification function

    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=1, hide_rest=True)
    #img_boundry1 = mark_boundaries(temp/255.0, mask)
    #img_boundry1 = mark_boundaries(temp / 2 + 0.5, mask)
    img_boundry1 = mark_boundaries(temp, mask)
    #plt.imshow(img_boundry1)
    im = Image.fromarray((img_boundry1 * 255).astype(np.uint8))
    #im.save(".\\results\\imageboundary_hide" + str(i) + ".jpeg")

    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=1, hide_rest=False)
    img_boundry2 = mark_boundaries(temp/255.0, mask)
    #plt.imshow(img_boundry2)
    im = Image.fromarray((img_boundry2 * 255).astype(np.uint8))
    transformToPIL = T.ToPILImage()
    transformed_input_image = transform(test_img)
    PILImage = transformToPIL(transformed_input_image)
    im = get_concat_h(PILImage, im)
    im.save(".\\results\\LIME_" + str(i) + ".jpeg")

x = 7