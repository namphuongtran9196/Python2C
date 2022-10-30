#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <stdio.h>
#include "yolov5.h"

using namespace cv;
using namespace std;

int main(int, char * argv[])
{
    Mat frame;
    //--- INITIALIZE VIDEOCAPTURE
    VideoCapture cap;
    // open the default camera using default API
    // cap.open(0);
    // OR advance usage: select any API backend
    int deviceID = 0;             // 0 = open default camera
    int apiID = cv::CAP_ANY;      // 0 = autodetect default API
    // open selected camera using selected API
    cap.open(deviceID, apiID);
    // check if we succeeded
    if (!cap.isOpened()) {
        cerr << "ERROR! Unable to open camera\n";
        return -1;
    }

    // TFlite model
    TFLiteModel *model = new TFLiteModel((char *)"../model.tflite");

    //--- GRAB AND WRITE LOOP
    cout << "Start grabbing" << endl
        << "Press any key to terminate" << endl;
    for (;;)
    {
        // wait for a new frame from camera and store it into 'frame'
        cap.read(frame);
        // check if we succeeded
        if (frame.empty()) {
            cerr << "ERROR! blank frame grabbed\n";
            break;
        }
        cvtColor(frame, frame, COLOR_BGR2RGB);
        letterbox(frame, frame, Size(640,640), 32, false);
        PredictResult res[MAX_OUTPUT];
        model->detect(frame,res);

        for (int i = 0; i < MAX_OUTPUT; i ++){
            if (res[i].score > 0.25){
                putText(frame, to_string(res[i].score), Point(res[i].xmin, res[i].ymin), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
                rectangle(frame, Point(res[i].xmin, res[i].ymin), Point(res[i].xmax, res[i].ymax), Scalar(0, 0, 255), 2);
            } else{
                break;
            }
        }
        // show live and wait for a key with timeout long enough to show images
        cvtColor(frame, frame, COLOR_RGB2BGR);
        imshow("Live", frame);
        if (waitKey(5) >= 0)
            break;
    }
    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}