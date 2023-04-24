import torch
import torchvision
import cv2
import numpy as np

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
threshold = 0.1
COMP_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BBoxGenerator:

    def __init__(self):
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval().to(COMP_DEVICE)

    def generate_bbox(self, frame, offset, single=True, returnImg=False):

        image = torch.as_tensor(frame, dtype=torch.float) / 255
        image = image.unsqueeze(0)
        image = image.permute(0, 3, 1, 2)

        # prediction
        with torch.no_grad():
            pred = self.model(image.to(COMP_DEVICE))

        pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in
                      list(pred[0]['labels'].cpu().detach().numpy())]  # Get the Prediction Score
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in
                      list(pred[0]['boxes'].cpu().detach().numpy())]  # Bounding boxes
        pred_score = list(pred[0]['scores'].cpu().detach().numpy())
        pred_t = [idx for idx, x in enumerate(pred_score) if
                  x > threshold]  # Get list of index with score greater than threshold.

        bboxes = []
        if pred_t:
            pred_t = pred_t[-1]
            pred_boxes = pred_boxes[:pred_t + 1]
            pred_class = pred_class[:pred_t + 1]
            pred_score = pred_score[:pred_t + 1]

            scores = []
            # get people
            for i, cls in enumerate(pred_class):
                if cls == "person":
                    bboxes.append(pred_boxes[i])
                    scores.append(pred_score[i])

            if single and len(scores) > 0:
                # select with the max prob
                scores = np.array(scores)
                maxIdx = np.argsort(scores)[-1]
                bboxes = [bboxes[maxIdx]]

        results = []
        for bbox in bboxes:

            # crop
            center = [(bbox[0][0] + bbox[1][0]) / 2, (bbox[0][1] + bbox[1][1]) / 2]
            center = [int(round(c)) for c in center]
            width = max(abs(bbox[0][0] - bbox[1][0]), abs(bbox[0][1] - bbox[1][1]))
            width = round(width)
            width = width + 1 if width % 2 == 1 else width
            width = int(width)

            # padding the image to prevent cropping out of index
            center = [c + width // 2 for c in center]
            if returnImg:
                temp = cv2.copyMakeBorder(frame.copy(), width // 2, width // 2, width // 2, width // 2,
                                          cv2.BORDER_CONSTANT,
                                          value=[0, 0, 0])

                result = temp[center[1] - width // 2: center[1] + width // 2,
                         center[0] - width // 2: center[0] + width // 2, :]

            # calculate pixel location from original image
            center = [c - width // 2 for c in center]
            center[0] -= offset[1]
            center[1] -= offset[0]

            topLeft_row = center[1] - width // 2
            topLeft_col = center[0] - width // 2

            if returnImg:
                results.append([result, [topLeft_col, topLeft_row, width, width]])
            else:
                results.append([topLeft_col, topLeft_row, width, width])

            if single:
                break

        return results

    def generate_batched_bboxes(self, frame, offset, single=True, returnImg=False):

        # Scaling
        images = torch.as_tensor(frame, dtype=torch.float) / 255
        # image = image.unsqueeze(0)
        images = images.permute(0, 3, 1, 2)

        # prediction
        with torch.no_grad():
            pred = self.model(images.to(COMP_DEVICE))

        batch_results = []

        for img in range(len(pred)):
            pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in
                          list(pred[img]['labels'].cpu().detach().numpy())]  # Get the Prediction Score
            pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in
                          list(pred[img]['boxes'].cpu().detach().numpy())]  # Bounding boxes
            pred_score = list(pred[img]['scores'].cpu().detach().numpy())
            pred_t = [idx for idx, x in enumerate(pred_score) if
                      x > threshold]  # Get list of index with score greater than threshold.

            bboxes = []
            if pred_t:
                pred_t = pred_t[-1]
                pred_boxes = pred_boxes[:pred_t + 1]
                pred_class = pred_class[:pred_t + 1]
                pred_score = pred_score[:pred_t + 1]

                scores = []
                # get people
                for i, cls in enumerate(pred_class):
                    if cls == "person":
                        bboxes.append(pred_boxes[i])
                        scores.append(pred_score[i])

                if single and len(scores) > 0:
                    # select with the max prob
                    scores = np.array(scores)
                    maxIdx = np.argsort(scores)[-1]
                    bboxes = [bboxes[maxIdx]]

            results = []
            for bbox in bboxes:

                # crop
                center = [(bbox[0][0] + bbox[1][0]) / 2, (bbox[0][1] + bbox[1][1]) / 2]
                center = [int(round(c)) for c in center]
                width = max(abs(bbox[0][0] - bbox[1][0]), abs(bbox[0][1] - bbox[1][1]))
                width = round(width)
                width = width + 1 if width % 2 == 1 else width
                width = int(width)

                # padding the image to prevent cropping out of index
                center = [c + width // 2 for c in center]
                if returnImg:
                    temp = cv2.copyMakeBorder(frame.copy(), width // 2, width // 2, width // 2, width // 2,
                                              cv2.BORDER_CONSTANT,
                                              value=[0, 0, 0])

                    result = temp[center[1] - width // 2: center[1] + width // 2,
                             center[0] - width // 2: center[0] + width // 2, :]

                # calculate pixel location from original image
                center = [c - width // 2 for c in center]
                center[0] -= offset[1]
                center[1] -= offset[0]

                topLeft_row = center[1] - width // 2
                topLeft_col = center[0] - width // 2

                if returnImg:
                    results.append([result, [topLeft_col, topLeft_row, width, width]])
                else:
                    results.append([topLeft_col, topLeft_row, width, width])

                if single:
                    break

            batch_results.append(results)
        del images
        return batch_results

    @staticmethod
    def check_ImgSize(img):
        '''
            make image size at least 800x800
        '''

        paddingHeight = 800 - min(img.shape[0], 800)
        paddingHeight = paddingHeight + 1 if paddingHeight % 2 == 1 else paddingHeight

        paddingWidth = 800 - min(img.shape[1], 800)
        paddingWidth = paddingWidth + 1 if paddingWidth % 2 == 1 else paddingWidth

        img = cv2.copyMakeBorder(img, paddingHeight // 2, paddingHeight // 2, paddingWidth // 2, paddingWidth // 2,
                                 cv2.BORDER_CONSTANT, value=[0, 0, 0])

        return img, [paddingHeight // 2, paddingWidth // 2]
