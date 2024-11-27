
#include "soextern.h"

image make_empty_image(int w, int h, int c)
{
    image out;
    out.data = 0;
    out.h = h;
    out.w = w;
    out.c = c;
    return out;
}

image make_image(int w, int h, int c){
    image out = make_empty_image(w,h,c);
    out.data = (float*)calloc(h * w * c, sizeof(float));
    return out;
}

void copy_image_from_bytes(image img, char *pdata){
    int w = img.w;
    int h = img.h;
    int c = img.c;
    int i, k, j;

    for (k = 0; k < c; ++k) {
        for (j = 0; j < h; ++j) {
            for (i = 0; i < w; ++i) {
                int index = k + c * i + c * w*j;
                float val = pdata[index];
                img.data[index] = val;
            }
        }
    }
}

cv::Mat convert_image_to_mat(image img){
    cv::Mat mat = cv::Mat(img.h, img.w, CV_8UC(img.c));
    int i, k, j;
    for (k = 0; k < img.c; ++k) {
        for (j = 0; j < img.h; ++j) {
            for (i = 0; i < img.w; ++i) {
                int index = k + img.c * i + img.c * img.w*j;
                float val = img.data[index];
                mat.data[index] = (unsigned char)val;
            }
        }
    }
    return mat;
}

void free_image(image img){ 
    if(img.data){
        free(img.data);
    }
}


// Load network (get instance)
InferYaml *load_network(const char *config_path, const int device_id)
{
    YAML::Node config = YAML::LoadFile(config_path);
    YAML::Node frame_config = config["FrameInfer"];
    InferYaml *net = new InferYaml(config, device_id);
    return net;
}

ImageResult* detect_img(InferYaml *model, image img){
    cv::Mat frame = convert_image_to_mat(img);
    // std::cout<< "("<<frame.cols<< ", "<<frame.rows<<")"<<std::endl;
    std::vector<DetectBox> detections = model->detect_img(frame);
    ImageResult *results = (ImageResult*)calloc(1, sizeof(ImageResult));
    int num_detections = detections.size();
    results[0].num_boxes = num_detections;
    DetectBox* boxes = (DetectBox*)calloc(num_detections, sizeof(DetectBox));
    for(int box_id=0; box_id<num_detections; ++box_id){
        boxes[box_id] = detections[box_id];
        // std::cout<< detections[box_id].xmin<<" "<<detections[box_id].ymin<<" "
        //     <<detections[box_id].xmax<<" "<<detections[box_id].ymax<<std::endl;
    }
    results[0].boxes = boxes;
    
    // std::cout<< "results num_detections: "<<results->num_boxes<<std::endl;
    return results;
}

void release(InferYaml *model){
    model->release();
    delete model;
}