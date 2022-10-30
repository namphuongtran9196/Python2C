#include "yolov5.h"

////////////////////////////////////////////////// Image processing //////////////////////////////////////////////////

void letterbox(Mat src, Mat& dst, Size new_shape, int stride, bool auto_mode, Scalar color,  bool scaleFill, bool scaleup ){
    // Resize and pad image while meeting stride-multiple constraints
    Size shape = src.size();

    // Scale ratio (new / old)
    float r = min((float) new_shape.width / (float) shape.width, (float) new_shape.height / (float) shape.height);
    
    if (!scaleup){ // only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0f);
    }

    // # Compute padding
    float ratio[2] = {r, r};  // width, height ratios
    int new_unpad[2] = {(int)round((float) shape.width * r), (int)round((float)shape.height * r)};
    int dw = new_shape.width - new_unpad[0];  // w padding
    int dh = new_shape.height - new_unpad[1];  // h padding

    if (auto_mode){ // minimum rectangle
        dw %= stride; // w padding
        dh %= stride; // h padding
    }
    else if (scaleFill){  // stretch
        dw = 0;  // w padding
        dh = 0;  // h padding
        new_unpad[0] = new_shape.width;
        new_unpad[1] = new_shape.height;
        ratio[0] = float(new_shape.width) / (float) shape.width;
        ratio[1] = float(new_shape.height) / (float) shape.height;
    }

    float dw_f = (float) dw / 2.0f;  // divide padding into 2 sides
    float dh_f = (float) dh / 2.0f;

    if (shape.width != new_unpad[0] or shape.height != new_unpad[1]){
        resize(src, dst, Size(new_unpad[0], new_unpad[1]), 0, 0, INTER_LINEAR);
    }
        
    int top = int(round(dh_f - 0.1));
    int bottom = int(round(dh_f + 0.1));
    int left = int(round(dw_f - 0.1));
    int right = int(round(dw_f + 0.1));

    copyMakeBorder(dst, dst, top, bottom, left, right, BORDER_CONSTANT, color);  // add border
}

////////////////////////////////////////////////// Tensorflow lite //////////////////////////////////////////////////

TFLiteModel::TFLiteModel(const char *model, long modelSize) {
	if (modelSize > 0) {
		initDetectionModel(model, modelSize);
	} 
}

TFLiteModel::TFLiteModel(const char *model) {
	initDetectionModel(model);
}
void TFLiteModel::initDetectionModel(const char *tfliteModel) {

	// Copy to model bytes as the caller might release this memory while we need it (EXC_BAD_ACCESS error on ios)
	m_model = tflite::FlatBufferModel::BuildFromFile(tfliteModel);
	assert(m_model != nullptr);

	// Build the interpreter
	tflite::ops::builtin::BuiltinOpResolver resolver;
	tflite::InterpreterBuilder builder(*m_model, resolver);
	builder(&m_interpreter);
	assert(m_interpreter != nullptr);

	// Allocate tensor buffers.
	assert(m_interpreter->AllocateTensors() == kTfLiteOk);
	assert(m_interpreter->Invoke() == kTfLiteOk);

	m_interpreter->SetNumThreads(1);
}

void TFLiteModel::initDetectionModel(const char *tfliteModel, long modelSize) {

	// Copy to model bytes as the caller might release this memory while we need it (EXC_BAD_ACCESS error on ios)
	m_modelBytes = (char *) malloc(sizeof(char) * modelSize);
	memcpy(m_modelBytes, tfliteModel, sizeof(char) * modelSize);
	m_model = tflite::FlatBufferModel::BuildFromBuffer(m_modelBytes, modelSize);
	assert(m_model != nullptr);

	// Build the interpreter
	tflite::ops::builtin::BuiltinOpResolver resolver;
	tflite::InterpreterBuilder builder(*m_model, resolver);
	builder(&m_interpreter);
	assert(m_interpreter != nullptr);

	// Allocate tensor buffers.
	assert(m_interpreter->AllocateTensors() == kTfLiteOk);
	assert(m_interpreter->Invoke() == kTfLiteOk);

	m_interpreter->SetNumThreads(1);
}

void *TFLiteModel::detect(Mat input, PredictResult res[]) {

    // convert the input image to float32
    input.convertTo(input, CV_32FC3);

	// allocate the tflite tensor
	m_interpreter->AllocateTensors();

	// get input & output layer of tflite model
	float *inputLayer = m_interpreter->typed_input_tensor<float>(0);
	float *outputLayer = m_interpreter->typed_output_tensor<float>(0);

	float *input_ptr = input.ptr<float>(0);

	// copy the input image to input layer
	memcpy(inputLayer, input_ptr, input.size().width * input.size().height * input.channels() * sizeof(float));

	// compute model instance
	if (m_interpreter->Invoke() != kTfLiteOk) {
		printf("Error invoking detection model");
	} else{
		for (int i = 0; i < MAX_OUTPUT; ++i) {
			res[i].xmin = outputLayer[i*5];
			res[i].ymin = outputLayer[i*5+1];
			res[i].xmax = outputLayer[i*5+2];
			res[i].ymax = outputLayer[i*5+3];
			res[i].score = outputLayer[i*5+4];
		}
	}
}