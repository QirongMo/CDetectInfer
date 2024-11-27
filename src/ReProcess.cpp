#pragma once
#include"ReProcess.h"


// RestorePadAndResize
RestorePadAndResize::RestorePadAndResize(float scale_x, float scale_y, int pad_left, int pad_top) {
    this->scale_x = scale_x;
    this->scale_y = scale_y;
    this->pad_left = pad_left;
    this->pad_top = pad_top;
}
void RestorePadAndResize::run(std::vector<DetectBox>& boxes) {
    for (auto box = boxes.begin(); box != boxes.end(); ++box) {
        box->xmin = (box->xmin - this->pad_left) / this->scale_x;
        box->ymin = (box->ymin - this->pad_top) / this->scale_y;
        box->xmax = (box->xmax - this->pad_left) / this->scale_x;
        box->ymax = (box->ymax - this->pad_top) / this->scale_y;
    }
}

// TransAndThresh
TransAndThresh::TransAndThresh(std::map<std::string, std::string> class_trans, std::map<std::string, float> class_threshes){
    this->class_trans = class_trans;
    this->class_threshes = class_threshes;
}
void TransAndThresh::run(std::vector<DetectBox>& boxes) {
    for (auto box = boxes.begin(); box != boxes.end();) {
        std::string class_name = box->class_name;
        float confidence = box->confidence;
        if (this->class_threshes.find(class_name) != this->class_threshes.end()) {
            // 
            if (confidence < this->class_threshes[class_name]) {
                // 
                box = boxes.erase(box);
                continue; //
            }
        }
        if (this->class_trans.find(class_name) != this->class_trans.end()) {
            //
            box->class_name = this->class_trans[class_name];
        }
        ++box;
    }
}


//int main(){
//    struct DetectBox box;
//    box.xmin = 10; box.ymin = 10; box.ymax = 20; box.xmax = 20, box.class_name="test", box.confidence=0.25;
//    struct DetectBox box2("hhh", 50, 50, 150, 150, 0.8);
//    std::vector<DetectBox> result = { box };
//    std::map<std::string, float> threshes;
//    std::map<std::string, std::string> class_trans;
//    class_trans.insert(std::make_pair("test", "new"));
//    //class_trans.insert(std::make_pair("hhh", "ttt"));
//    threshes.insert(std::make_pair("test", 0.9));
//    TransAndThresh tran_thresh(class_trans, threshes);
//    tran_thresh.run(result);
//    printf("class_name: %s\n", result[0].class_name);
//    int m = result.size();
//    printf("size: %d", m);
//
//}