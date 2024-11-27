
#include "TrtUtils.h"




void Logger::log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept {
    if (severity != Severity::kINFO) {
        std::cout << msg << std::endl;
    }
}


char* serialized_file(const char* model_path, size_t &mdsize){
    // 读取文件
    std::ifstream ifile(model_path, std::ios::in | std::ios::binary);
    if (!ifile) {
        std::cout << "read serialized file failed\n";
        std::abort();
    }
    ifile.seekg(0, std::ios::end);
    mdsize = ifile.tellg();
    ifile.clear();
    ifile.seekg(0, std::ios::beg);
    char* serialized_engine = new char[mdsize];
    ifile.read(serialized_engine, mdsize);
    ifile.close();
    std::cout << "Read serialized file finished! model size: " << mdsize << std::endl;
    return serialized_engine;
}


size_t get_dim_size(nvinfer1::Dims dims){
    size_t size = 1;
    for(int i=0; i<dims.nbDims; i++)
        size *= dims.d[i];
    return size;
}