#pragma once
#include"PreProcessYaml.h"
#include"ReProcessYaml.h"
#include"FrameInfer.h"

class InferYaml {
private:
	std::vector<PreProcessYaml> preprocess_ops;
	std::vector<ReProcessYaml*> reprocess_ops;
	FrameInfer* model;
public:
	InferYaml(YAML::Node config, int device_id=0) {
		//
		YAML::Node preprocess_list = config["PreProcess"];
		for (size_t i = 0; i < preprocess_list.size(); ++i) {
			YAML::Node r_cfg = preprocess_list[i];
			PreProcessYaml op = GetPreprocess(r_cfg);
			this->preprocess_ops.push_back(op);
		}
		//
		YAML::Node reprocess_list = config["ReProcess"];
		for (size_t i = 0; i < reprocess_list.size(); ++i) {
			YAML::Node r_cfg = reprocess_list[i];
			ReProcessYaml* op = GetReprocess(r_cfg);
			this->reprocess_ops.push_back(op);
		}
		// 
		// std::string model_path = "/home/mqr/Desktop/cvdetect/Models/helmet/helmet20230922.onnx";
		// std::vector<std::string> class_names = { "head", "helmet" };
		YAML::Node frame_config = config["FrameInfer"];
		this->model = GetFrameInfer(frame_config, device_id);
		// this->model = new YoloV5OnnxInfer(frame_config, device_id);
	}
	std::vector<DetectBox> run(float* blob) {
		return this->model->run(blob);
	};
	std::vector<DetectBox> detect_img(const cv::Mat img) {
		ImgInfo* img_info = new ImgInfo();
		cv::Mat input_img;
		img.copyTo(input_img);
		for (std::vector<PreProcessYaml>::iterator op = preprocess_ops.begin(); op != preprocess_ops.end(); ++op) {
			input_img = op->run(input_img, img_info);
		}
		std::vector<DetectBox> detections = this->model->detect_img(input_img);
		for (std::vector<ReProcessYaml*>::iterator op = reprocess_ops.begin(); op != reprocess_ops.end(); ++op) {
			(*op)->run(detections, img_info);
			
		}
		return detections;
	};
	void release(){
		model->release();
		delete model;
		model = nullptr;
	}
	
};