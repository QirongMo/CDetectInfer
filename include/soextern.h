
#pragma once
#include"InferYaml.h"


typedef struct image {
    int w;
    int h;
    int c;
    float *data;
} image;


extern "C"{
    image make_image(int w, int h, int c);
    void copy_image_from_bytes(image img, char *pdata);
    void free_image(image img);

    InferYaml *load_network(const char *config_path, const int device_id);
    ImageResult* detect_img(InferYaml *model, image img);
    void release(InferYaml *model);
}

