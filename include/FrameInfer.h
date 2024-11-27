#pragma once
#include <yaml-cpp/yaml.h>  
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include "utils.h"
// tensorrt
#include "TrtUtils.h"


class FrameInfer {
public:
	std::vector<std::string> class_names;
	float conf_thresh, iou_thresh;
	int device_id;
	YAML::Node config;

	FrameInfer(YAML::Node &config, int device_id=0);

	virtual std::vector<DetectBox> run(float* blob);

	virtual std::vector<DetectBox> detect_img(const cv::Mat img);

	virtual void release();

};

//---------YoloV5OnnxInfer----------------
class YoloV5OnnxInfer: public FrameInfer{
private:
	Ort::Env env{ nullptr };
	Ort::Session *session;
	Ort::SessionOptions *session_options;
	Ort::AllocatorWithDefaultOptions allocator;

	OrtCUDAProviderOptions *cuda_options;

	std::vector<const char*> inputNodeNames;
	std::vector<const char*> outputNodeNames;
	int input_h, input_w;
public:
	YoloV5OnnxInfer(YAML::Node &config, int device_id=0);
	std::vector<DetectBox> run(float* blob);
	std::vector<DetectBox> detect_img(const cv::Mat img);
	void release();
	std::vector<DetectBox> decoder_result(const std::vector<Ort::Value>& outputTensors);
};

//--------Yolov5TrtInfer----------------
class Yolov5TrtInfer: public FrameInfer{
private:
    // 创建logger：日志记录器
    Logger mlogger;
    // runtime
    nvinfer1::IRuntime* runtime = nullptr;
    // engine
    nvinfer1::ICudaEngine* engine = nullptr;
    // context
    nvinfer1::IExecutionContext* context = nullptr;
    // stream
    cudaStream_t stream = nullptr;
    // buffers
    std::vector<BuffersInfos> buffers_info;
    std::vector<void*> buffers;
public:
	Yolov5TrtInfer(YAML::Node &config, int device_id=0);
	std::vector<DetectBox> run(float* blob);
	std::vector<DetectBox> detect_img(const cv::Mat img);
	void release();
	std::vector<DetectBox> decoder_result(float* out); 
};

FrameInfer* GetFrameInfer(YAML::Node config, int device_id);
