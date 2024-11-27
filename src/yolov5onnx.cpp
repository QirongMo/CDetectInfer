
#include"FrameInfer.h"


std::vector<DetectBox> decoder_result(const std::vector<Ort::Value>& outputTensors, std::vector<std::string> class_names,
	float confThreshold = 0.25, float iouThreshold = 0.35);

YoloV5OnnxInfer::YoloV5OnnxInfer(YAML::Node &config, int device_id): FrameInfer(config, device_id=device_id){
	this->env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "yolov5onnx");
	// std::cout<<"开始加载模型"<<std::endl;
	// session_options
	session_options = new Ort::SessionOptions();
	// printf("device_id: %d\n", device_id);
	if(device_id >= 0){
		cuda_options = new OrtCUDAProviderOptions();
		cuda_options->device_id = device_id;
		cuda_options->cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;
		session_options->AppendExecutionProvider_CUDA(*cuda_options);
	}
	
	// session
	std::string model_path = config["model"].as<std::string>();
	session = new Ort::Session(this->env, model_path.c_str(), *session_options);

	// names
	char* input_name = session->GetInputName(0, allocator);
	char* output_name = session->GetOutputName(0, allocator);
	this->inputNodeNames.push_back(input_name);
	this->outputNodeNames.push_back(output_name);
	// shape
	Ort::TypeInfo inputTypeInfo = session->GetInputTypeInfo(0);
	std::vector<int64_t> inputTensorShape = inputTypeInfo.GetTensorTypeAndShapeInfo().GetShape(); // [num_batch, 3, input_h, input_w]
	this->input_h = inputTensorShape[2], this->input_w = inputTensorShape[3];
	// class_names
	// this->class_names = class_names;
	// 
	// this->conf_thresh = conf_thresh, this->iou_thresh = iou_thresh;
	// std::cout<<"模型加载成功"<<std::endl;

}
std::vector<DetectBox> YoloV5OnnxInfer::run(float* blob) {
	std::array<int64_t, 4> input_shape_info{ 1, 3, this->input_h, this->input_w};
	auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Ort::Value input_tensor_ = Ort::Value::CreateTensor<float>(allocator_info, blob,
		this->input_h * this->input_w * 3, input_shape_info.data(), input_shape_info.size());
	std::vector<Ort::Value> ort_outputs;
	//
	ort_outputs = session->Run(Ort::RunOptions{ nullptr },
		this->inputNodeNames.data(), & input_tensor_, this->inputNodeNames.size(),
		this->outputNodeNames.data(), this->outputNodeNames.size());
	// nms
	return decoder_result(ort_outputs);
}

std::vector<DetectBox> YoloV5OnnxInfer::detect_img(cv::Mat img) {
	float* blob = BlobFromImage(img);
	std::vector<DetectBox> detections = this->run(blob);
	delete blob;
	return detections;
}


void YoloV5OnnxInfer::release(){
	Ort::OrtRelease(*session);
	session = nullptr;

	Ort::OrtRelease(env);

	if(device_id >= 0){
		delete cuda_options;
		cuda_options = nullptr;
	}

	Ort::OrtRelease(session_options->release());
	session_options = nullptr;

	std::cout<<"释放模型"<<std::endl;	

}


std::vector<DetectBox> YoloV5OnnxInfer::decoder_result(const std::vector<Ort::Value>& outputTensors) {
	std::vector<cv::Rect> boxes;
	std::vector<float> confs;
	std::vector<int> classIds;
	auto* rawOutput = outputTensors[0].GetTensorData<float>();
	std::vector<int64_t> outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape(); // outputShape: [num_batch, num_anchor, 1+4+num_class]
	size_t count = outputTensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
	std::vector<float> output(rawOutput, rawOutput + count);
	int num_classes = outputShape[2] - 5;

	int elementsInBatch = (int)(outputShape[1] * outputShape[2]);
	// only for batch size = 1
	float max_conf = 0;
	for (auto it = output.begin(); it != output.begin() + elementsInBatch; it += outputShape[2])
	{
		float obj_conf = it[4];
		int bestClassId = -1;
		float bestConf = 0;
		for (int i = 5; i < num_classes + 5; i++)
		{
			if (it[i] > bestConf)
			{
				bestConf = it[i];
				bestClassId = i - 5;
			}
		}
		if (bestClassId == -1) continue;
		float confidence = bestConf * obj_conf;
		if (confidence > max_conf) max_conf = confidence;
		int centerX = (int)(it[0]);
		int centerY = (int)(it[1]);
		int width = (int)(it[2]);
		int height = (int)(it[3]);
		int left = centerX - width / 2;
		int top = centerY - height / 2;

		boxes.emplace_back(left, top, width, height);
		confs.emplace_back(confidence);
		classIds.emplace_back(bestClassId);
	}
	//printf("max_conf: %f\n", max_conf);
	std::vector<DetectBox> detections = Yolov5Nms(boxes, confs, classIds, class_names, conf_thresh, iou_thresh);
	return detections;
}