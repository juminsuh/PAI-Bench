import cv2
import os
import numpy as np
from PIL import Image


ATTRIBUTES = [
    'skin',
    'l_brow',
    'r_brow',
    'l_eye',
    'r_eye',
    'eye_g',
    'l_ear',
    'r_ear',
    'ear_r',
    'nose',
    'mouth', # 이빨 
    'u_lip', # 윗입술
    'l_lip', # 아랫입술
    'neck',
    'neck_l',
    'cloth',
    'hair',
    'hat'
]


COLOR_LIST = [
    [0, 0, 0],
    [255, 85, 0],
    [255, 170, 0],
    [255, 0, 85],
    [255, 0, 170],
    [0, 255, 0],
    [85, 255, 0],
    [170, 255, 0],
    [0, 255, 85],
    [0, 255, 170],
    [0, 0, 255],
    [85, 0, 255],
    [170, 0, 255],
    [0, 85, 255],
    [0, 170, 255],
    [255, 255, 0],
    [255, 255, 85],
    [255, 255, 170],
    [255, 0, 255],
]


def vis_parsing_maps(image, segmentation_mask, save_image=False, cs_path="", bm_path="", filename=""):
    # Create numpy arrays for image and segmentation mask
    image = np.array(image).copy().astype(np.uint8)
    segmentation_mask = segmentation_mask.copy().astype(np.uint8)
    blended_image_save_path = os.path.join(cs_path, filename) # ./assets/colored_segmentation_output/BrunoMars/0.jpg
    number = os.path.splitext(filename)[0]
    
    # Create a color mask
    segmentation_mask_color = np.zeros((segmentation_mask.shape[0], segmentation_mask.shape[1], 3))

    num_classes = np.max(segmentation_mask)

    mask_path = os.path.join(bm_path, number) # ./assets/binary_mask_output/celeb
    os.makedirs(mask_path, exist_ok=True)

    for class_index in range(1, num_classes + 1):
        class_pixels = np.where(segmentation_mask == class_index) # find every pixel belongs to class_index
        segmentation_mask_color[class_pixels[0], class_pixels[1], :] = COLOR_LIST[class_index]

        # save binary mask of each region 
        if class_index in [1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13]:
            binary_mask = (segmentation_mask == class_index)
            mask_image = (binary_mask * 255).astype(np.uint8)
            img = Image.fromarray(mask_image)
            mask_save_path = os.path.join(mask_path, f"{class_index}.jpg") # ./assets/binary_mask_output/BrunoMars/0/l_eye
            img.save(mask_save_path)
        
    segmentation_mask_color = segmentation_mask_color.astype(np.uint8)

    # Convert image to BGR format for blending
    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Blend the image with the segmentation mask
    blended_image = cv2.addWeighted(bgr_image, 0.6, segmentation_mask_color, 0.4, 0)

    # Save the result if required
    if save_image:
        cv2.imwrite(blended_image_save_path, blended_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    return blended_image
