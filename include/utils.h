
#pragma once
#include<iostream>
#include <opencv2/opencv.hpp>


struct DetectBox {
    float xmin, ymin, xmax, ymax;
    float confidence;
    int class_id;
    std::string class_name;
    DetectBox() {}
    DetectBox(int class_id_, std::string name, float xmin_, float ymin_, float xmax_, float ymax_, float conf = 0.0) {
        class_id = class_id_;
        class_name = name;
        xmin = xmin_; ymin = ymin_; xmax = xmax_; ymax = ymax_;
        confidence = conf;
    }
};

struct ImageResult {
    int num_boxes;
    DetectBox* boxes;
};

float* BlobFromImage(cv::Mat& iImg);


std::vector<DetectBox> Yolov5Nms(const std::vector<cv::Rect> &boxes, const std::vector<float> &confs,
	const std::vector<int> &classIds, const std::vector<std::string> &class_names, 
    const float confThreshold, const float iouThreshold);