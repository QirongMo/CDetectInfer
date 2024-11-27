

import sys
from ctypes import *

import cv2
import numpy as np
import yaml

lib = CDLL("./build/libdetectinfer.so", RTLD_GLOBAL)

class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

copy_image_from_bytes = lib.copy_image_from_bytes
copy_image_from_bytes.argtypes = [IMAGE,c_char_p]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

free_image = lib.free_image
free_image.argtypes = [IMAGE]

load_network = lib.load_network
load_network.argtypes = [c_char_p, c_int]
load_network.restype = c_void_p

release_net = lib.release
release_net.argtypes = [c_void_p]


class DetectBox(Structure):
    _fields_ = [("xmin", c_float),
                ("ymin", c_float),
                ("xmax", c_float),
                ("ymax", c_float),
                ("confidence", c_float),
                ('class_id', c_int),
                ("class_name", c_char_p)]

class ImageResult(Structure):
    _fields_ = [("num_boxes", c_int),
                ("boxes", POINTER(DetectBox))]

detect_img = lib.detect_img
detect_img.argtypes = [c_void_p, IMAGE]
detect_img.restype = POINTER(ImageResult)


def main():
    yaml_path = "./samples/yolov10.yaml"
    network = load_network(yaml_path.encode())

    import cv2
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    while True:
        ret, img = cap.read()
        img_h, img_w = img.shape[:2]
        darknet_image = make_image(img_w, img_h, 3)
        copy_image_from_bytes(darknet_image, img.tobytes())
        results = detect_img(network, darknet_image)[0]
        free_image(darknet_image)
        num_boxes = results.num_boxes
        print("num_boxes: ", num_boxes)
        boxes = results.boxes
        for i in range(num_boxes):
            print(boxes[i].xmin, boxes[i].ymin, boxes[i].xmax, boxes[i].ymax)
            x1 = int(boxes[i].xmin)
            y1 = int(boxes[i].ymin)
            x2 = int(boxes[i].xmax)
            y2 = int(boxes[i].ymax)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255),5)
            # print('conf', boxes[i].confidence, 'class_id: ', boxes[i].cls_id)
        cv2.imshow("frame", img)
        # cv2.imwrite("result.jpg", img)
        # break
        if cv2.waitKey(int(1000/30)) == ord('q'):
            break
    cv2.destroyWindow("frame")
    release_net(network)
    del network

def main2():
    yaml_path = "./samples/yolov10.yaml"
    network = load_network(yaml_path.encode(), 0)

    import cv2
    import os
    from pathlib import Path
    from tqdm import tqdm
    root = "/home/mqr/Pictures"
    allfiles = list(os.listdir(root))
    # for _ in range(1000):
    for file in tqdm(allfiles):
        if Path(file).suffix.lower() not in ['.jpg','.png','.jpeg']:
            continue
        img_path = os.path.join(root, file)
        img = cv2.imread(img_path)
        img_h, img_w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        darknet_image = make_image(img_w, img_h, 3)
        copy_image_from_bytes(darknet_image, img.tobytes())
        results = detect_img(network, darknet_image)[0]
        free_image(darknet_image)
        num_boxes = results.num_boxes
        # print("num_boxes: ", num_boxes)
        boxes = results.boxes
        # for i in range(num_boxes):
        #     print(boxes[i].xmin, boxes[i].ymin, boxes[i].xmax, boxes[i].ymax)
        #     x1 = int(boxes[i].xmin)
        #     y1 = int(boxes[i].ymin)
        #     x2 = int(boxes[i].xmax)
        #     y2 = int(boxes[i].ymax)
        #     cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255),5)
        #     # print('conf', boxes[i].confidence, 'class_id: ', boxes[i].cls_id)
        # cv2.imshow("frame", img)
    release_net(network)
    del network        


if __name__ == "__main__":
    main()