#include <iostream>  
#include <yaml-cpp/yaml.h>  


void read_op(YAML::Node config) {
    YAML::const_iterator it = config.begin();
    std::string op_name = it->first.as<std::string>();
    YAML::Node op_config = it->second;
    YAML::Node target_size = op_config["target_size"];
    if (target_size) {
        if (target_size.IsSequence()) {
            printf("target_size: [%d, %d]", target_size[0].as<int>(), target_size[1].as<int>());
        }
        else {
            printf("target_size: [%d, %d]", target_size.as<int>(), target_size[1].as<int>());
        }
    }
}

//
//int main() {
//    std::string config_path = "C://Users/mqr/Pictures/model.yaml";
//    YAML::Node config = YAML::LoadFile(config_path);
//
//   /* if (config["ip"]) {
//        std::string ip = config["ip"].as<std::string>();
//        printf("ip: %s\n", ip);
//    }
//    else {
//        std::cout << "Key1 does not exist!\n";
//    }
//    if (config["port"]) {
//        int port = config["port"].as<int>();
//        printf("port: %d", port);
//    }
//    else {
//        std::cout << "Key1 does not exist!\n";
//    }*/
//    // 
//    YAML::Node preprocess_list = config["PreProcess"];
//    if (!preprocess_list) return 0;
//
//    //  
//    for (size_t i = 0; i < preprocess_list.size(); ++i) {
//        YAML::Node r_cfg = preprocess_list[i];
//        read_op(r_cfg);
//        break;
//    }
// 
//  /*  YAML::Node r_cfg = pp_config[0];
//    if (r_cfg.IsMap()) {
//        printf("���Ǹ��ֵ�\n");   
//        read_op(r_cfg);
//    }
//    else if (r_cfg.IsSequence()) {
//        printf("");
//    }
//    else {
//        printf("");
//    }
//    */
//
//    return 0;
//}