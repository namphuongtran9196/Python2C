#include <math.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"


#include <iostream>

using namespace cv;
using namespace std;

struct PredictResult {
	float score = 0;
	float ymin = 0.0;
	float xmin = 0.0;
	float ymax = 0.0;
	float xmax = 0.0;
};

const int MAX_OUTPUT = 1000;

class TFLiteModel {
public:
	TFLiteModel(const char *model, long modelSize);
	TFLiteModel(const char *model);
    ~TFLiteModel();
    void *detect(Mat src, PredictResult *res);
	void initDetectionModel(const char *model);
	void initDetectionModel(const char *model, long modelSize);
private:
	// members
	bool m_hasDetectionModel = false;
	char *m_modelBytes = nullptr;
	std::unique_ptr<tflite::FlatBufferModel> m_model;
	std::unique_ptr<tflite::Interpreter> m_interpreter;
};

void letterbox(Mat src, Mat& dst, Size new_shape = Size(640,640), int stride=32, bool auto_mode = true, 
				Scalar color= Scalar(114,114,114),  bool scaleFill=false, bool scaleup=true );