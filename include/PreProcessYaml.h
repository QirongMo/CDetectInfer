#pragma once
#include "PreProcess.h"
#include <yaml-cpp/yaml.h>  

class PreProcessYaml {
public:
	PreProcessOP* op;
	PreProcessYaml(YAML::Node config){}
	cv::Mat run(cv::Mat img, ImgInfo* img_info) {
		return this->op->run(img, img_info);
	}
};


class RGBReverseYaml:public PreProcessYaml {
private:
	void init_ins(YAML::Node config);
public:
	RGBReverseYaml(YAML::Node config):PreProcessYaml(config){
		this->init_ins(config);
	};
	
};


class ResizeImgYaml:public PreProcessYaml {
private:
	void init_ins(YAML::Node config);
public:
	ResizeImgYaml(YAML::Node config):PreProcessYaml(config){
		this->init_ins(config);
	}
};


class MaxshapeResizeYaml:public PreProcessYaml {
private:
	void init_ins(YAML::Node config);
public:
	MaxshapeResizeYaml(YAML::Node config):PreProcessYaml(config){
		this->init_ins(config);
	};

};


class NormalizeImageYaml: public PreProcessYaml {
private:
	void init_ins(YAML::Node config);
public:
	NormalizeImageYaml(YAML::Node config): PreProcessYaml(config){
		this->init_ins(config);
	};

};

PreProcessYaml GetPreprocess(YAML::Node config);
