#include <opencv2/objdetect.hpp>
#include <opencv2/xobjdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <string>
#include <fstream>
#include <stdio.h>

using namespace std;
using namespace cv;





int main(void)
{
   
    vector<cv::String> path_positive;
    glob("C:\\Users\\jacop_\\Desktop\\computer_vision_final_project\\POSITIVE_SAMPLE\\*.png", path_positive, false);

	vector<cv::String> path_negative;
	glob("C:\\Users\\jacop_\\Desktop\\computer_vision_final_project\\NEGATIVE_SAMPLE\\*.jpg", path_negative, false);

	Mat src_gray;
	for (int i = 0; i < path_positive.size(); i++) {
		Mat dest;
		Mat src = imread(path_positive[i], IMREAD_COLOR);
		cvtColor(src, src_gray, COLOR_BGR2GRAY); 
		
		string path;
		path = "C:\\Users\\jacop_\\Desktop\\computer_vision_final_project\\POSITIVE_BW\\";
		path.append("img");
		path.append(to_string(i));
		path.append(".jpg");
		imwrite(path, src_gray);
	}


	for (int i = 0; i < path_negative.size(); i++) {
		Mat dest;
		Mat src = imread(path_negative[i], IMREAD_COLOR);
		cvtColor(src, src_gray, COLOR_BGR2GRAY);
		
		string path;
		path = "C:\\Users\\jacop_\\Desktop\\computer_vision_final_project\\NEGATIVE_BW\\";
		path.append("img");
		path.append(to_string(i));
		path.append(".jpg");
		imwrite(path, src_gray);
	}
	

    return 0;
}
