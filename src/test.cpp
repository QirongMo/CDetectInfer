// #include <onnxruntime_cxx_api.h>
// #include <opencv2/opencv.hpp>
// #include"PreProcess.h"
// #include"ReProcess.h"
// #include"InferYaml.h"
#include "soextern.h"

void test() {
	std::string config_path = "/home/mqr/Desktop/CDetectInfer/Models/helmet/helmet.yaml";
	// YAML::Node config = YAML::LoadFile(config_path);
	// InferYaml model(config);
	InferYaml *model = load_network(config_path.c_str(), 0);
	std::string img_path = "/home/mqr/Pictures/vlcsnap-2024-05-14-11h32m43s226.png";
	cv::Mat img = cv::imread(img_path);
	// cv::resize(img, img, cv::Size(640, 640));
	std::vector<DetectBox> detections = model->detect_img(img);
	for (std::vector<DetectBox>::iterator box = detections.begin(); box != detections.end(); ++box) {
		cv::rectangle(img,
			cv::Point(box->xmin, box->ymin), cv::Point(box->xmax, box->ymax),
			cv::Scalar(229, 160, 21), 5);
		cv::imshow("result", img);
		//cv::waitKey(0);
	}
	cv::imwrite("result.jpg", img);
}



int main() {
	test();
}
