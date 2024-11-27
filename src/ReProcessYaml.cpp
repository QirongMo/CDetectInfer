#pragma once
#include "ReProcessYaml.h"


RestorePadAndResizeYaml::RestorePadAndResizeYaml(YAML::Node config) : ReProcessYaml(config) {
    this->op = new RestorePadAndResize();
}

void RestorePadAndResizeYaml::run(std::vector<DetectBox>& boxes, ImgInfo* img_info) {
    delete this->op;
    this->op = new RestorePadAndResize(img_info->scale_x, img_info->scale_y, img_info->pad_left, img_info->pad_top);
    this->op->run(boxes);

}


TransAndThreshYaml::TransAndThreshYaml(YAML::Node config): ReProcessYaml(config){
    std::map<std::string, std::string> class_trans;
    std::map<std::string, float> class_threshes;
    for (const auto& pair : config) {
        std::string class_name = pair.first.as<std::string>();
        YAML::Node op_config = pair.second;
        //printf("class_name: %s ", class_name);
        YAML::Node trans_name = op_config["trans_name"];
        if (trans_name) {
            class_trans.insert(std::make_pair(class_name, trans_name.as<std::string>()));
            //printf("trans_name: %s ", trans_name.as<std::string>());
        }
        YAML::Node thresh = op_config["thresh"];
        if (thresh) {
            class_threshes.insert(std::make_pair(class_name, thresh.as<float>()));
            //printf("thresh: %f ", thresh.as<float>());
        }
        //printf("\n");
    }
    this->op = new TransAndThresh(class_trans, class_threshes);
}

ReProcessYaml* GetReprocess(YAML::Node config) {
    YAML::const_iterator it = config.begin();
    std::string op_name = it->first.as<std::string>();
    YAML::Node op_config = it->second;
    if (op_name == "RestorePadAndResizeData")
        return new RestorePadAndResizeYaml(op_config);
    else if (op_name == "TransAndThreshData")
        return new TransAndThreshYaml(op_config);
    else
        return new ReProcessYaml(op_config);
}

//int main(){
//    struct DetectBox box;
//    box.xmin = 10; box.ymin = 10; box.ymax = 20; box.xmax = 20, box.class_name="excavator", box.confidence=0.25;
//    struct DetectBox box2("bulldozer", 50, 50, 150, 150, 0.8);
//    std::vector<DetectBox> result = { box, box2 };
//    // ∂¡»°yaml
//	std::string config_path = "C://Users/mqr/Pictures/configs.yaml";
//	YAML::Node config = YAML::LoadFile(config_path);
//    ImgInfo img_info;
//    TransAndThreshYaml tran_thresh(config, img_info);
//    tran_thresh.run(result);
//    printf("class_name: %s\n", result[0].class_name);
//    int m = result.size();
//    printf("size: %d", m);
//
//}