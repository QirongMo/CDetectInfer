
#include <iostream>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <fstream>
#include <vector>

#ifndef CUDA_CHECK
#define CUDA_CHECK(callstr)\
    {\
        cudaError_t error_code = callstr;\
        if (error_code != cudaSuccess) {\
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__;\
            assert(0);\
        }\
    }
#endif  // CUDA_CHECK


class Logger : public nvinfer1::ILogger {
 public:
  void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override;
};

char* serialized_file(const char* model_path, size_t &mdsize);


size_t get_dim_size(nvinfer1::Dims dims);

struct BuffersInfos{
    int32_t binding_id;
    int32_t dim_size;
    nvinfer1::Dims dims;
    nvinfer1::DataType data_type;
    bool is_input;
};
