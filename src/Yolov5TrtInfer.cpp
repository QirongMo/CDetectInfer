
#include "FrameInfer.h"



std::vector<DetectBox> decoder_result(const std::vector<Ort::Value>& outputTensors, std::vector<std::string> class_names,
	float confThreshold = 0.25, float iouThreshold = 0.35);

Yolov5TrtInfer::Yolov5TrtInfer(YAML::Node &config, int device_id): FrameInfer(config, device_id=device_id){
    if(device_id<0){
        std::cout<<"device_id需大于0"<<std::endl; 
        std::abort();
    }
    cudaSetDevice(device_id);
    // 创建runtime
    runtime = nvinfer1::createInferRuntime(mlogger);
    // assert(runtime != nullptr);
    // 反序列化模型，得到engine
    std::string model_path = config["model"].as<std::string>();
    // std::string model_path = "/home/mqr/Desktop/CDetectInfer/Models/helmet/helmet20230922.trt";
    size_t size = 0;
    char* serialized_engine = serialized_file(model_path.c_str(), size);
    // 创建engine
    engine = runtime->deserializeCudaEngine(serialized_engine, size, nullptr);
    delete[] serialized_engine;
    // 创建context
    context = engine->createExecutionContext();
    // 创建buffers
    int32_t num_bindings = engine->getNbBindings();
    for(int32_t i=0; i<num_bindings; i++){
        BuffersInfos buffer_info;
        void* temp_buffs;
        const char* name = engine->getBindingName(i);
        buffer_info.dims = context->getTensorShape(name);;
        buffer_info.dim_size = get_dim_size(buffer_info.dims);
        buffer_info.is_input = engine->bindingIsInput(i);
        buffer_info.data_type = engine->getBindingDataType(i);
        CUDA_CHECK(cudaMalloc(&temp_buffs, buffer_info.dim_size*sizeof(float)));
        buffers_info.emplace_back(buffer_info);
        buffers.emplace_back(temp_buffs);
        // delete temp_buffs;
    }
}

std::vector<DetectBox> Yolov5TrtInfer::run(float* blob) {
    // 拷贝输入到显存
    CUDA_CHECK(cudaMemcpyAsync(buffers[0], blob, buffers_info[0].dim_size*sizeof(float), cudaMemcpyHostToDevice, stream));
    // 执行推理
    context->enqueueV2(&buffers[0], stream, nullptr);
    // 将输出从显存拷贝到内存
    float* out = new float[buffers_info[1].dim_size];
    CUDA_CHECK(cudaMemcpyAsync(out, buffers[1], buffers_info[1].dim_size*sizeof(float), cudaMemcpyDeviceToHost, stream));
    // 解析输出
    std::vector<DetectBox> detections = decoder_result(out); 
    delete out;
    return detections;
}

std::vector<DetectBox> Yolov5TrtInfer::detect_img(cv::Mat img) {
	float* blob = BlobFromImage(img);
	std::vector<DetectBox> detections = this->run(blob);
	delete blob;
	return detections;
}

void Yolov5TrtInfer::release(){
    cudaStreamSynchronize(stream);
    // 清理buffers
    for(int i=0; i<buffers.size(); i++){
        cudaFree(buffers[i]);
    }
    buffers.clear();
    buffers_info.clear();
    // 释放stream
    cudaStreamDestroy(stream);
    // 释放context
    context->destroy();
    context = nullptr;
    // 是否engine
    engine->destroy();
    engine = nullptr;
    // runtime
    runtime->destroy();
    runtime=nullptr;

	std::cout<<"模型已释放"<<std::endl;	

}

std::vector<DetectBox> Yolov5TrtInfer::decoder_result(float* out){
    // 输出的大小
    int num_anchors = buffers_info[1].dims.d[1];
    int out_dim = buffers_info[1].dims.d[2];
    int num_classes = out_dim-5;
    if(num_classes != class_names.size()){
        std::cout<<"类别不一致，模型类别数量为："<<num_classes<<" 但给的类别数量为："<<class_names.size()<<std::endl;
        std::abort();
    }
    // 分解模型输出
	std::vector<cv::Rect> boxes;
	std::vector<float> confs;
	std::vector<int> classIds;
    // 按anchor遍历所有输出
    float max_conf = 0;
    for (int it = 0; it < num_anchors; it++)
    {
        float obj_conf = out[out_dim*it+4];
        int bestClassId = -1;
        float bestConf = 0;
        for (int class_id=0; class_id<num_classes; class_id++)
        {
            float conf = out[out_dim*it+5+class_id];
            if (conf> bestConf){
                bestConf = conf;
                bestClassId = class_id;
            }
        }
        if (bestClassId == -1) continue;
        float confidence = bestConf * obj_conf;
        if (confidence > max_conf) max_conf = confidence;
        int centerX = (int)(out[out_dim*it]);
        int centerY = (int)(out[out_dim*it+1]);
        int width = (int)(out[out_dim*it+2]);
        int height = (int)(out[out_dim*it+3]);
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