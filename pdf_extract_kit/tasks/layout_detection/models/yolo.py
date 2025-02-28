import os
import cv2
import torch
import numpy as np
from pdf_extract_kit.registry import MODEL_REGISTRY
from pdf_extract_kit.utils.visualization import visualize_bbox
from pdf_extract_kit.dataset.dataset import ImageDataset

@MODEL_REGISTRY.register('layout_detection_yolo')
class LayoutDetectionYOLO:
    def __init__(self, config):
        """
        Initialize the LayoutDetectionYOLO class.

        Args:
            config (dict): Configuration dictionary containing model parameters.
        """
        # Mapping from class IDs to class names
        self.id_to_names = {
            0: 'title', 
            1: 'plain text',
            2: 'abandon', 
            3: 'figure', 
            4: 'figure_caption', 
            5: 'table', 
            6: 'table_caption', 
            7: 'table_footnote', 
            8: 'isolate_formula', 
            9: 'formula_caption'
        }

        # Load the YOLO model from the specified path
        try:
            from doclayout_yolo import YOLOv10
            self.model = YOLOv10(config['model_path'])
        except AttributeError:
            from ultralytics import YOLO
            self.model = YOLO(config['model_path'])

        # Set model parameters
        self.img_size = config.get('img_size', 1280)
        self.conf_thres = config.get('conf_thres', 0.25)
        self.iou_thres = config.get('iou_thres', 0.45)
        self.visualize = config.get('visualize', False)
        self.nc = config.get('nc', 10)
        self.workers = config.get('workers', 8)
        self.device = config.get('device', 'cpu')
        
        if self.iou_thres > 0:
            import torchvision
            self.nms_func = torchvision.ops.nms

    def predict(self, images, result_path, image_ids=None):
        """
        Predict formulas in images.

        Args:
            images (list): List of images to be predicted.
            result_path (str): Path to save the prediction results.
            image_ids (list, optional): List of image IDs corresponding to the images.

        Returns:
            list: List of prediction results.
        """
        results = []
        for idx, image in enumerate(images):
            result = self.model.predict(image, imgsz=self.img_size, conf=self.conf_thres, iou=self.iou_thres, verbose=False, device=self.device)[0]
            if self.visualize:
                if not os.path.exists(result_path):
                    os.makedirs(result_path)
                boxes = result.__dict__['boxes'].xyxy
                classes = result.__dict__['boxes'].cls
                scores = result.__dict__['boxes'].conf

                if self.iou_thres > 0:
                    indices = self.nms_func(boxes=torch.Tensor(boxes), scores=torch.Tensor(scores),iou_threshold=self.iou_thres)
                    boxes, scores, classes = boxes[indices], scores[indices], classes[indices]
                    if len(boxes.shape) == 1:
                        boxes = np.expand_dims(boxes, 0)
                        scores = np.expand_dims(scores, 0)
                        classes = np.expand_dims(classes, 0)

                # Determine the base name of the image
                if image_ids:
                    base_name = image_ids[idx]
                else:
                    base_name = os.path.splitext(os.path.basename(image))[0]

                # Create subdirectories for each class if they don't exist
                diagrams_path = os.path.join(result_path, 'diagrams')
                formulas_path = os.path.join(result_path, 'formulas')
                tables_path = os.path.join(result_path, 'tables')
                os.makedirs(diagrams_path, exist_ok=True)
                os.makedirs(formulas_path, exist_ok=True)
                os.makedirs(tables_path, exist_ok=True)


                # Filter results based on desired classes (figure, table, isolate_formula)
                filtered_boxes = []
                filtered_classes = []
                filtered_scores = []
                for i in range(len(boxes)):
                    if int(classes[i]) in [3, 5, 8]:  # figure, table, isolate_formula
                        filtered_boxes.append(boxes[i])
                        filtered_classes.append(classes[i])
                        filtered_scores.append(scores[i])

                # Save each filtered element as a separate image in respective folders
                for i, box in enumerate(filtered_boxes):
                    x1, y1, x2, y2 = map(int, box)
                    cropped_image = np.array(image)[y1:y2, x1:x2]
                    class_name = self.id_to_names[int(filtered_classes[i])]
                    page_number = base_name.split('_page_')[1].split('_')[0] # Extract page number from filename
                    label_text = f"{class_name.capitalize()} Page {page_number} - {i}"

                    # Add label text below the image
                    text_to_draw = label_text
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    font_thickness = 1
                    text_color = (0, 0, 0)  # Black color
                    text_size, _ = cv2.getTextSize(text_to_draw, font, font_scale, font_thickness)
                    text_height = text_size[1]
                    image_height, image_width, _ = cropped_image.shape
                    labeled_image = np.full((image_height + text_height + 10, image_width, 3), 255, dtype=np.uint8) # White background
                    labeled_image[:image_height, :, :] = cropped_image # Copy image to top

                    text_x = 0
                    text_y = image_height + text_height + 5 # Position text below image

                    cv2.putText(labeled_image, text_to_draw, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)


                    result_name = f"{base_name}_{class_name}_{i}.png"
                    
                    if class_name == 'figure':
                        save_path = diagrams_path
                    elif class_name == 'isolate_formula':
                        save_path = formulas_path
                    elif class_name == 'table':
                        save_path = tables_path
                    else:
                        save_path = result_path #fallback just in case

                    cv2.imwrite(os.path.join(save_path, result_name), labeled_image)
            
            results.append(result) #keep the original result
        return results
