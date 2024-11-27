#pragma once
#include<map>
#include<vector>
#include"utils.h"


//PreProcessOP
class ReProcessOP {
public:
    ReProcessOP() {}
    virtual void run(std::vector<DetectBox>& boxes) = 0;
};

// RestorePadAndResize
class RestorePadAndResize: public ReProcessOP {
private:
    float scale_x, scale_y;
    int pad_left, pad_top;
public:
    RestorePadAndResize(float scale_x = 1.0, float scale_y = 1.0, int pad_left = 0, int pad_top = 0);
    void run(std::vector<DetectBox>& boxes);
};


// TransAndThresh
class TransAndThresh : public ReProcessOP {
private:
    std::map<std::string, std::string> class_trans;
    std::map<std::string, float> class_threshes;
public:
    TransAndThresh(std::map<std::string, std::string> class_trans, std::map<std::string, float> class_threshes);
    void run(std::vector<DetectBox>& boxes);
};


