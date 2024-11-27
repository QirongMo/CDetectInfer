#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>


struct ImgInfo {
    float scale_x;
    float scale_y;
    int pad_left;
    int pad_top;
    ImgInfo(float scale_x = 1.0, float scale_y = 1.0, int pad_left = 0, int pad_top = 0) : scale_x(scale_x),
        scale_y(scale_y), pad_left(pad_left), pad_top(pad_top) {}
};

//PreProcessOP
class PreProcessOP {
public:
    PreProcessOP(){}
    virtual cv::Mat run(cv::Mat img, ImgInfo* img_info) = 0;
};

// RGBReverse BGR2BGR
class RGBReverse: public PreProcessOP{
public:
    RGBReverse(){};
    cv::Mat run(cv::Mat img, ImgInfo* img_info);
};

// MaxshapeResize 最大边resize
class MaxshapeResize : public PreProcessOP {
private:
    int max_shape;
    int interp;
public:
    MaxshapeResize(int max_shape = 0, int interp = cv::INTER_LINEAR){
        this->max_shape = max_shape;
        this->interp = interp;
    }
    cv::Mat run(cv::Mat img, ImgInfo* img_info);
};

// ResizeImg resize
class ResizeImg : public PreProcessOP {
private:
    int target_w, target_h;
    bool keep_ratio;
    int interp;
    int pad_r, pad_g, pad_b;

    void get_target(int target = 0, int target_w = 0, int target_h = 0);
    void get_pad(int pad = 114, int pad_r = 114, int pad_g = 114, int pad_b = 114);
    void generate_scale(int org_w, int org_h, float& scale_x, float& scale_y);
public:
    ResizeImg(int target = 0, int target_w = 0, int target_h = 0, bool keep_ratio = false, int interp = cv::INTER_LINEAR,
        int pad = 114, int pad_r = 114, int pad_g = 114, int pad_b = 114) {
        this->get_target(target, target_w, target_h);
        this->keep_ratio = keep_ratio;
        this->interp = interp;
        this->get_pad(pad, pad_r, pad_g, pad_b);
        //printf("target_w: %d, target_h: %d\n", this->target_w, this->target_h);
    }
    cv::Mat run(cv::Mat img, ImgInfo* img_info);
};

class NormalizeImage : public PreProcessOP {
private:
    bool is_scale;
    std::vector<float> mean;
    std::vector<float> std_;
    void init_mean(float mean = 0, float mean_r = 0, float mean_g = 0, float mean_b = 0);
    void init_std(float std_ = 0, float std_r = 1, float std_g = 1, float std_b = 1);
public:
    NormalizeImage(float mean = 0, float mean_r = 0, float mean_g = 0, float mean_b = 0,
        float std_ = 1.0, float std_r = 1.0, float std_g = 1.0, float std_b = 1.0, bool is_scale = true) {
        this->init_mean(mean, mean_r, mean_g, mean_b);
        this->init_std(std_, std_r, std_g, std_b);
        this->is_scale = is_scale;
    }
    cv::Mat run(cv::Mat img, ImgInfo* img_info);
};
