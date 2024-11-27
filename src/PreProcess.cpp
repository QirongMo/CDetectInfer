

#include "PreProcess.h"


//---RGBReverse---
cv::Mat RGBReverse::run(cv::Mat img, ImgInfo* img_info) {
    cv::Mat tg_img;
    cv::cvtColor(img, tg_img, cv::COLOR_RGB2BGR);
    return tg_img;
}
// ---MaxshapeResize---

cv::Mat MaxshapeResize::run(cv::Mat img, ImgInfo* img_info) {
    //
    int width = img.cols, height = img.rows;
    // 
    int maxDimension = std::max(width, height);
    // 
    float scale = static_cast<float>(this->max_shape) / maxDimension;
    cv::Mat tg_img;
    cv::resize(img, tg_img, cv::Size(), scale, scale, this->interp);
    img_info->scale_x *= scale;
    img_info->scale_y *= scale;
    return tg_img;
}

// ---ResizeImg---

void ResizeImg::get_target(int target, int target_w, int target_h) {
    assert(target != 0 || (target_w != 0 && target_h != 0));
    if (target_w != 0 && target_h != 0) {
        this->target_w = target_w;
        this->target_h = target_h;
    }
    else {
        this->target_w = target;
        this->target_h = target;
    }
}

void ResizeImg::get_pad(int pad, int pad_r, int pad_g, int pad_b) {
    if (pad_r != 114 or pad_g != 114 or pad_b != 114) {
        this->pad_r = pad_r;
        this->pad_g = pad_g;
        this->pad_b = pad_b;
    }
    else {
        this->pad_r = pad;
        this->pad_g = pad;
        this->pad_b = pad;
    }
}
void ResizeImg::generate_scale(int org_w, int org_h, float& scale_x, float& scale_y) {
    float sx = static_cast<float>(this->target_w) / org_w;
    float sy = static_cast<float>(this->target_h) / org_h;
    if (this->keep_ratio) {
        scale_x = scale_y = std::min(sx, sy);
    }
    else {
        scale_x = sx;
        scale_y = sy;
    }

}

cv::Mat ResizeImg::run(cv::Mat img, ImgInfo* img_info)  {
    //printf("run ResizeImg\n");
    // 
    int width = img.cols, height = img.rows;
    // 
    float scale_x, scale_y;
    this->generate_scale(width, height, scale_x, scale_y);
    // 
    cv::Mat tg_img;
    cv::resize(img, tg_img, cv::Size(), scale_x, scale_y, this->interp);
    //
    int new_width = tg_img.cols, new_height = tg_img.rows;
    // 
    // printf("new_width: %d, new_height: %d\n", new_width, new_height);
    int pad_left = (this->target_w - new_width) / 2, pad_top = (this->target_h - new_height) / 2;
    int pad_right = this->target_w - new_width - pad_left, pad_bottom = this->target_h - new_height - pad_top;
    // 
    cv::copyMakeBorder(tg_img, tg_img, pad_top, pad_bottom, pad_left, pad_right,
        cv::BORDER_CONSTANT, cv::Scalar(this->pad_b, this->pad_g, this->pad_r));
    // img_info
    img_info->scale_x *= scale_x;
    img_info->scale_y *= scale_y;
    img_info->pad_left = pad_left;
    img_info->pad_top = pad_top;
    //cv::imshow("result", tg_img);
    // cv::imwrite("result.jpg", image);
    //cv::waitKey(0);
    return tg_img;
}

//---NormalizeImage----

void NormalizeImage::init_mean(float mean, float mean_r, float mean_g, float mean_b) {
    if (mean_r == 0 && mean_g == 0 && mean_b != 0) {
        mean_r = mean_g = mean_b = mean;
    }
    this->mean = { mean_r, mean_g, mean_b };
}
void NormalizeImage::init_std(float std_, float std_r, float std_g, float std_b) {
    if (std_r == 1 && std_g == 1 && std_b != 1) {
        std_r = std_g = std_b = std_;
    }
    if (std_r == 0 || std_g == 0 || std_b == 0) {
        throw "std_r, std_g, std_b不能为0";
    };
    this->std_ = { std_r, std_g, std_b };
}

cv::Mat NormalizeImage::run(cv::Mat img, ImgInfo* img_info) {
    img.convertTo(img, CV_32FC3);
    float e = 1.0;
    if (this->is_scale) {
        e = 1.0 / 255;
    }
    cv::Mat tg_img;
    img.convertTo(tg_img, CV_32FC3);
    for (int h = 0; h < tg_img.rows; h++) {
        for (int w = 0; w < tg_img.cols; w++) {
            // printf("value: %f, mean: %f, std: %f, after: %f\n", tg_img.at<cv::Vec3f>(h, w)[0], this->mean[2], 
            //    this->std_[2], (tg_img.at<cv::Vec3f>(h, w)[0] * e - this->mean[2]) / this->std_[2]);
            tg_img.at<cv::Vec3f>(h, w)[0] =
                (tg_img.at<cv::Vec3f>(h, w)[0] * e - this->mean[2]) / this->std_[2];
            tg_img.at<cv::Vec3f>(h, w)[1] =
                (tg_img.at<cv::Vec3f>(h, w)[1] * e - this->mean[1]) / this->std_[0];
            tg_img.at<cv::Vec3f>(h, w)[2] =
                (tg_img.at<cv::Vec3f>(h, w)[2] * e - this->mean[0]) / this->std_[0];
        }
    }
    return tg_img;
}


//int main(){
//    ImgInfo img_info;
//    //printf("scale_x: %f\n", img_info.scale_x);
//    std::string img_path = "C:/Users/mqr/Pictures/0a64feaf_b86f_4982_82f4_421dd7784f45.JPG";
//
//    //
//    cv::Mat image = cv::imread(img_path);
//
//    ResizeImg resize_img(0, 640, 640, true);
//    image = resize_img.run(image, &img_info);
//    cv::imshow("Image", image);
//    cv::waitKey(0);
//    printf("pad_left: %d, pad_top: %d\n", img_info.pad_left, img_info.pad_top);
//
//}