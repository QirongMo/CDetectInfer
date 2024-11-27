//#include <iostream>  
//#include <yaml-cpp/yaml.h>  
//#include "PreProcess.h"
//#include <opencv2/opencv.hpp>
#include "PreProcessYaml.h"
#include <yaml-cpp/yaml.h>  

void RGBReverseYaml::init_ins(YAML::Node config){
	this->op = new RGBReverse();
}

void ResizeImgYaml::init_ins(YAML::Node config) {
	int target = 0, target_w = 0, target_h = 0;
	bool keep_ratio = false;
	int interp = cv::INTER_LINEAR;
	int pad = 114, pad_r = 114, pad_g = 114, pad_b = 114;
	YAML::Node size_node = config["target_size"];
	if (size_node) {
		if (size_node.IsSequence()) {
			target_w = size_node[0].as<int>();
			target_h = size_node[1].as<int>();
		}
		else {
			target = size_node.as<int>();
		}
	}
	if (config["keep_ratio"]) {
		keep_ratio = config["keep_ratio"].as<bool>();
	}
	if (config["interp"]) {
		interp = config["interp"].as<int>();
	}
	YAML::Node pad_node = config["pad_color"];
	if (pad_node) {
		if (pad_node.IsSequence()) {
			pad_r = pad_node[0].as<int>();
			pad_g = pad_node[1].as<int>();
			pad_b = pad_node[2].as<int>();
		}
		else {
			pad = pad_node.as<int>();
		}
	}
	this->op = new ResizeImg(target, target_w, target_h, keep_ratio, interp, pad = 114, pad_r, pad_g, pad_b);
}


void MaxshapeResizeYaml::init_ins(YAML::Node config) {
	int max_shape, interp = 2;
	max_shape = config["max_shape"].as<int>();
	if (config["interp"]) {
		interp = config["interp"].as<int>();
	}
	this->op = new MaxshapeResize(max_shape, interp);
}

// NormalizeImageYaml
void NormalizeImageYaml::init_ins(YAML::Node config) {
	float mean = 0, mean_r = 0, mean_g = 0, mean_b = 0;
	float std_ = 1.0, std_r = 1.0, std_g = 1.0, std_b = 1.0;
	bool is_scale = true;
	YAML::Node mean_node = config["mean"];
	if (mean_node) {
		if (mean_node.IsSequence()) {
			mean_r = mean_node[0].as<float>();
			mean_g = mean_node[1].as<float>();
			mean_b = mean_node[2].as<float>();
		}
		else {
			mean = mean_node.as<float>();
		}
	}
	YAML::Node std_node = config["mean"];
	if (std_node) {
		if (std_node.IsSequence()) {
			std_r = std_node[0].as<float>();
			std_g = std_node[1].as<float>();
			std_b = std_node[2].as<float>();
		}
		else {
			std_ = std_node.as<float>();
		}
	}
	this->op = new NormalizeImage(mean, mean_r, mean_g, mean_b, std_, std_r, std_g, std_b, is_scale);
}

PreProcessYaml GetPreprocess(YAML::Node config) {
	YAML::const_iterator it = config.begin();
	std::string op_name = it->first.as<std::string>();
	YAML::Node op_config = it->second;
	if (op_name == "ResizeInput")
		return ResizeImgYaml(op_config);
	else if (op_name == "NormalizeInput")
		return NormalizeImageYaml(op_config);
	else if (op_name == "RGBReverseInput")
		return RGBReverseYaml(op_config);
	else if (op_name == "MaxshapeResize")
		return MaxshapeResizeYaml(op_config);
	else
		return PreProcessYaml(op_config);
}

//int main(){
//    ImgInfo img_info;
//    printf("scale_x: %f\n", img_info.scale_x);
//    std::string img_path = "C:/Users/mqr/Pictures/0a64feaf_b86f_4982_82f4_421dd7784f45.JPG";
//    // ∂¡»° ‰»ÎÕºœÒ
//    cv::Mat image = cv::imread(img_path);
//	// ∂¡»°yaml
//	std::string config_path = "C://Users/mqr/Pictures/configs.yaml";
//	YAML::Node config = YAML::LoadFile(config_path);
//
//
//    ResizeImgYaml resize_img(config);
//    image = resize_img.run(image, &img_info);
//    cv::imshow("Image", image);
//    cv::waitKey(0);
//    printf("scale_x: %f, pad_top: %d\n", img_info.scale_x, img_info.pad_top);
//
//}