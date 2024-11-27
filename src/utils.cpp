
#include "utils.h"


float* BlobFromImage(cv::Mat& iImg) {
	int img_c = iImg.channels();
	int img_h = iImg.rows;
	int img_w = iImg.cols;
	float* iBlob = new float[img_h * img_w * img_c];
	for (int c = 0; c < img_c; c++) {
		for (int h = 0; h < img_h; h++) {
			for (int w = 0; w < img_w; w++) {
				/*iBlob[c * img_w * img_h + h * img_w + w] = iImg.at<cv::Vec3b>(h, w)[c]/255.0f;*/
				iBlob[c * img_w * img_h + h * img_w + w] = iImg.at<cv::Vec3f>(h, w)[c];
			}
		}
	}
	return iBlob;
}


std::vector<DetectBox> Yolov5Nms(const std::vector<cv::Rect> &boxes, const std::vector<float> &confs,
	const std::vector<int> &classIds, const std::vector<std::string> &class_names, 
    const float confThreshold, const float iouThreshold){
    std::vector<int> indices;
	cv::dnn::NMSBoxes(boxes, confs, confThreshold, iouThreshold, indices);
    std::vector<DetectBox> detections;
	for (int idx : indices)
	{
		DetectBox det;
		det.xmin = boxes[idx].x;
		det.ymin = boxes[idx].y;
		det.xmax = boxes[idx].x + boxes[idx].width;
		det.ymax = boxes[idx].y + boxes[idx].height;
		det.confidence = confs[idx];
		int classId = classIds[idx];
		det.class_id = classId;
		det.class_name = class_names[classId];
		detections.emplace_back(det);
	}
    return detections;
}