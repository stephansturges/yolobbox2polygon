import os
import glob
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import requests
import shutil
#from segment_anything import sam_model_registry, SamPredictor
import re
from segment_anything_hq import sam_model_registry, SamPredictor
import gc
import random


def enhance_image_to_HDR(img):
    # Convert the image from BGR to LAB
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # Split the LAB image into different channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to the L-channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    
    # Merge the CLAHE enhanced L-channel with the A and B channels
    limg = cv2.merge((cl,a,b))
    
    # Convert the image back to BGR
    img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    return img
def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)
def show_mask(mask, ax, random_color=False, color=None):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    elif color:
        color = np.concatenate([np.array(color), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
      
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  

def show_boxes_on_image(raw_image, boxes):
    plt.figure(figsize=(10,10))
    plt.imshow(raw_image)
    for box in boxes:
      show_box(box, plt.gca())
    plt.axis('on')
    plt.show()

def show_points_on_image(raw_image, input_points, input_labels=None):
    plt.figure(figsize=(10,10))
    plt.imshow(raw_image)
    input_points = np.array(input_points)
    if input_labels is None:
      labels = np.ones_like(input_points[:, 0])
    else:
      labels = np.array(input_labels)
    show_points(input_points, labels, plt.gca())
    plt.axis('on')
    plt.show()

def show_points_and_boxes_on_image(raw_image, boxes, input_points, input_labels=None):
    plt.figure(figsize=(10,10))
    plt.imshow(raw_image)
    input_points = np.array(input_points)
    if input_labels is None:
      labels = np.ones_like(input_points[:, 0])
    else:
      labels = np.array(input_labels)
    show_points(input_points, labels, plt.gca())
    for box in boxes:
      show_box(box, plt.gca())
    plt.axis('on')
    plt.show()


def show_points_and_boxes_on_image(raw_image, boxes, input_points, input_labels=None):
    plt.figure(figsize=(10,10))
    plt.imshow(raw_image)
    input_points = np.array(input_points)
    if input_labels is None:
      labels = np.ones_like(input_points[:, 0])
    else:
      labels = np.array(input_labels)
    show_points(input_points, labels, plt.gca())
    for box in boxes:
      show_box(box, plt.gca())
    plt.axis('on')
    plt.show()


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_masks_on_image(raw_image, masks, scores):
    if len(masks.shape) == 4:
      masks = masks.squeeze()
    if scores.shape[0] == 1:
      scores = scores.squeeze()

    nb_predictions = scores.shape[-1]
    fig, axes = plt.subplots(1, nb_predictions, figsize=(15, 15))

    for i, (mask, score) in enumerate(zip(masks, scores)):
      mask = mask.cpu().detach()
      axes[i].imshow(np.array(raw_image))
      show_mask(mask, axes[i])
      axes[i].title.set_text(f"Mask {i+1}, Score: {score.item():.3f}")
      axes[i].axis("off")
    plt.show()


def getLabels(labelPath):
    with open(labelPath) as f:
        # Preparing list for annotation of BB (bounding boxes)
        labels = []
        for line in f:
            labels += [line.rstrip()]

    return labels

def readLabelBB(labels, w, h):
    parsedLabels = []
    for i in range(len(labels)):
        bb_current = labels[i].split()
        objClass = bb_current[0]
        x_center, y_center = int(float(bb_current[1]) * w), int(float(bb_current[2]) * h)
        box_width, box_height = int(float(bb_current[3]) * w), int(float(bb_current[4]) * h)
        parsedLabels.append((x_center, y_center, box_width, box_height))
    return parsedLabels, objClass

def getConvertedBoxes(labels, image_width, image_height, grow_factor=1):
    converted_boxes = []
    class_ids = []
    for i in range(len(labels)):
        bb_current = labels[i].split()
        class_id = int(bb_current[0])
        x_center, y_center = float(bb_current[1]), float(bb_current[2])
        box_width , box_height = float(bb_current[3]), float(bb_current[4])
        
        box_width = box_width * grow_factor
        box_height = box_height * grow_factor

        # Convert to top left and bottom right coordinates
        x0 = int((x_center - box_width / 2) * image_width)
        y0 = int((y_center - box_height / 2) * image_height)
        x1 = int((x_center + box_width / 2) * image_width)
        y1 = int((y_center + box_height / 2) * image_height)
        class_ids.append(class_id)
        converted_boxes.append([x0, y0, x1, y1])
    return  class_ids, converted_boxes



def main():
    print("PyTorch version:", torch.__version__)
    print("Torchvision version:", torchvision.__version__)
    print("CUDA is available:", torch.cuda.is_available())

    
    bbox_dataset = './yolo_dataset/'
    seg_dataset = './segmentation_dataset'
    image_files = glob.glob(f"./{bbox_dataset}/**/*", recursive=True)
    image_files = [f for f in image_files if f.lower().endswith(('.jpg', '.png'))]


    print("Number of images:", len(image_files))
    # iterate through each image file and add it to a tuple 
    image_labels = []
    for imgPath in image_files:
        # get the label file path
        labelPath = re.sub(r'\.[^.]*$', '.txt', imgPath)

        # replace images with labels
        labelPath = labelPath.replace("images", "labels")
        # add the image and label path to a tuple
        image_labels.append((imgPath, labelPath))


    model_type = "vit_h" #"vit_l/vit_b/vit_h/vit_tiny"
    sam_checkpoint = "sam_hq_vit_h.pth"
    device = "cpu"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)

    sam.to(device=device)

    predictor = SamPredictor(sam)

    device = "cuda"

    samgpu = sam_model_registry[model_type](checkpoint=sam_checkpoint)

    samgpu.to(device=device)
    predictorgpu = SamPredictor(samgpu)
        

    # random.seed(42)  # Setting a random seed to ensure reproducibility
    # random.shuffle(image_lables)  # Randomizing the image_labels array

    for imgPath, labelPath in image_labels:
        destination = f'{seg_dataset}'
        # if label file is in destination folder then skip
        label_file = imgPath.split('/')[-1].split('.')[0]
        seg_label_path = os.path.join(destination, f'{label_file}.txt')
        if os.path.exists(seg_label_path):
            label_file = imgPath.split('/')[-1].split('.')[0]
            print(f'{label_file} already exists in {destination}')
            continue
        labels = getLabels(labelPath)
        print("image name:")
        print(imgPath)


        if labels == []:
            pass
        else:
            raw_image = Image.open(imgPath).convert("RGB")

            image = cv2.imread(imgPath, cv2.IMREAD_COLOR)
            image = enhance_image_to_HDR(image)
            image = cv2.GaussianBlur(image, (15, 15), 0)
            image = adjust_gamma(image, 1.4)
            image = cv2.resize(image, None, fx=0.5, fy=0.5)
            image = cv2.add(image, np.array([30.0])) # up brightness by 30
            

            try:
                predictorgpu.set_image(image)
                h, w = image.shape[:2]
                class_ids, bounding_boxes = getConvertedBoxes(labels, w, h, grow_factor=1)
                #show_boxes_on_image(raw_image, bounding_boxes) 
                input_boxes = torch.tensor(bounding_boxes, device=predictorgpu.device)
                transformed_boxes = predictorgpu.transform.apply_boxes_torch(input_boxes, image.shape[:2])
                

                masks, _, low_rez_first_iteration_logits = predictorgpu.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes,
                    multimask_output=False,
                )
                print("ran inference on GPU")
            except Exception as e:
                print("Failed inference on GPU with: ")
                print(e)
                continue
                predictor.set_image(image)
                h, w = image.shape[:2]
                class_ids, bounding_boxes = getConvertedBoxes(labels, w, h, grow_factor=1)
                #show_boxes_on_image(raw_image, bounding_boxes) 
                input_boxes = torch.tensor(bounding_boxes, device=predictor.device)

                transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
                masks, _, low_rez_first_iteration_logits = predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes,
                    multimask_output=False,
                )
                print("ran inference on CPU !!!")
                

            torch.cuda.empty_cache()
            gc.collect()


            # IMPORTANT: we are first creating a rough mask with the HQ model within the size of the bbox
            # then using that mask as a low rez input to a second iteration of the same model with a growth
            # factor set up on the bounding box. The reason for this is to "grab" things that are slighty
            # on the edge of the original bounding box. We are using a "single mask" return because
            # multi-mask seems to look for too much detail on the model, and doesn't perform as well
            # in testing... tbd whether this can be improved in the future!

            image = cv2.imread(imgPath, cv2.IMREAD_COLOR)
            image = enhance_image_to_HDR(image)
            image = cv2.GaussianBlur(image, (5, 5), 0)
            image = adjust_gamma(image, 1.4)
            
            try:
                predictorgpu.set_image(image)
                h, w = image.shape[:2]
                class_ids, bounding_boxes = getConvertedBoxes(labels, w, h, grow_factor=1)
                #show_boxes_on_image(raw_image, bounding_boxes) 
                input_boxes = torch.tensor(bounding_boxes, device=predictorgpu.device)
                transformed_boxes = predictorgpu.transform.apply_boxes_torch(input_boxes, image.shape[:2])
                masks, _, _ = predictorgpu.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes,
                    multimask_output=False,
                    mask_input=low_rez_first_iteration_logits,
                )

                print("ran inference on GPU")
            except Exception as e:
                print("Failed inference on GPU with: ")
                print(e)
                continue
                predictor.set_image(image)
                h, w = image.shape[:2]
                class_ids, bounding_boxes = getConvertedBoxes(labels, w, h, grow_factor=1)
                #show_boxes_on_image(raw_image, bounding_boxes) 
                input_boxes = torch.tensor(bounding_boxes, device=predictor.device)
                
                transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
                masks, _, _ = predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes,
                    multimask_output=False,
                    mask_input=low_rez_first_iteration_logits,
                )
                print("ran inference on CPU !!!!!!")
            


            print("made masks and there are X:")
            print(str(len(masks)))
            for i,mask in enumerate(masks):
                print("working on mask No: " + str(i))
                binary_mask = masks[i].squeeze().cpu().numpy().astype(np.uint8)
                contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                try:
                    # Sorting the contours by area and selecting the top 3
                    largest_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1] #sorted as if we take top X and stack but no longer doing this

                    # Merging the top 3 contours together
                    merged_contour = np.vstack(largest_contours)
                    segmentation = merged_contour.flatten().tolist()
                    mask = segmentation

                    # convert mask to numpy array of shape (N,2)
                    mask = np.array(mask).reshape(-1, 2)

                    # normalize the pixel coordinates
                    mask_norm = mask / np.array([w, h])
                    class_id = class_ids[i]
                    yolo = mask_norm.reshape(-1)

                    # if folder does not exist, create it
                    if not os.path.exists(destination):
                        os.makedirs(destination)
                
                except Exception as e:
                    print(str(e))
                    continue


                print(f'writing segmentation file in {label_file} to {destination}')
                # create labels folder if it does not exist
                # if not os.path.exists(os.path.join(destination, 'labels')):
                #     os.makedirs(os.path.join(destination, 'labels'))
                with open(seg_label_path, "a") as f:
                    for val in yolo:
                        print("{} {:.6f}".format(class_id,val))
                        f.write("{} {:.6f}".format(class_id,val))
                    f.write("\n")
    
            torch.cuda.empty_cache()
            gc.collect()

            # create images folder if it does not exist
            if not os.path.exists(os.path.join(destination, 'images')):
                os.makedirs(os.path.join(destination, 'images'))
            # copy image to destination/images
            # strip faulty extensions
            shutil.copy(imgPath, f'{destination}/images/{os.path.basename(os.path.splitext(os.path.splitext(imgPath)[0])[0]) + os.path.splitext(imgPath)[1]}')

if __name__ == "__main__":
    main()