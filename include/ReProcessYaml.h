#pragma once
#include "ReProcess.h"
#include <yaml-cpp/yaml.h>  
#include "PreProcess.h"


class ReProcessYaml {
public:
	ReProcessOP* op;
	ReProcessYaml(YAML::Node config) {}
	virtual void run(std::vector<DetectBox>& boxes, ImgInfo* img_info){
		//this->update_params(img_info);
		return this->op->run(boxes);
	}
	//virtual void update_params(ImgInfo* img_info)=0;
};


class RestorePadAndResizeYaml : public ReProcessYaml {
public:
	RestorePadAndResizeYaml(YAML::Node config);
	void run(std::vector<DetectBox>& boxes, ImgInfo* img_info);
};

class TransAndThreshYaml: public ReProcessYaml {
public:
	TransAndThreshYaml(YAML::Node config);
};

ReProcessYaml* GetReprocess(YAML::Node config);
