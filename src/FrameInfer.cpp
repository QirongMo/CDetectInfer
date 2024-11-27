
#include "FrameInfer.h"


FrameInfer::FrameInfer(YAML::Node &config, int device_id){
    this->config = config;
    this->device_id = device_id;
    // class_names
    YAML::Node label_list = config["label_list"];  
    for (const auto& label : label_list){
        class_names.push_back(label.as<std::string>());
    }
    // thresh
    if(config["thresh"].IsDefined())
        conf_thresh = config["thresh"].as<float>();
    else 
        conf_thresh = 0.25;	
    // nms_thresh
    if(config["nms_thresh"].IsDefined())
        iou_thresh = config["nms_thresh"].as<float>();	
    else 
        iou_thresh = 0.35;		
}

std::vector<DetectBox> FrameInfer::run(float* blob) {
		std::vector<DetectBox> detections;
		return detections;
};

std::vector<DetectBox> FrameInfer::detect_img(const cv::Mat img){
		std::vector<DetectBox> detections; 
		return detections; 
};

void FrameInfer::release(){};

FrameInfer* GetFrameInfer(YAML::Node config, int device_id){
	std::string frame_type = config["type"].as<std::string>();
	if(frame_type=="Yolov5OnnxInfer"){
		return new YoloV5OnnxInfer(config, device_id);
	}
    else if(frame_type=="Yolov5TrtInfer"){
        return new Yolov5TrtInfer(config, device_id);
    }
	return new FrameInfer(config, device_id);
}
