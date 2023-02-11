#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <string>
#include <fstream>
#include <stdio.h>
#include <opencv2/ml/ml.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>


using namespace cv;
using namespace std;
using namespace ml;



int main(int argc, const char** argv) {

/********************************************************************************************************************************/
			// initialization of the path in which recover image to process

	vector<cv::String> positive_crop_image_path;
	vector<cv::String> negative_image_path;

	cv::glob("C:\\Users\\jacop_\\Desktop\\computer_vision_final_project\\POSITIVE_BW\\*.jpg", positive_crop_image_path, false);
	cv::glob("C:\\Users\\jacop_\\Desktop\\computer_vision_final_project\\NEGATIVE_SAMPLE_bw\\*.jpg", negative_image_path, false);

	std::cout << "loading the images and extracting HOG feature descriptors... both for positive and negative samples\n" << std::endl;
/********************************************************************************************************************************/
	
	// parameters initialization for SVM training
	Mat HOGFeat_train(9032, 3780, CV_32FC1); // each row is a sample
	Mat labels_train;

	// parameters initialization for HOG descriptors
	vector<Point> Roi_locations;
	vector<float> Roi_descriptors;
	HOGDescriptor d1(Size(64, 128), Size(16, 16), Size(8, 8), Size(8, 8), 9, 1);

	//initialization of useful variables
	Mat input;
	int size_count = 0;
	Mat input_img;
	

/********************************************************************************************************************************/

	for (int y = 0; y < positive_crop_image_path.size(); y++) {

		input_img = imread(positive_crop_image_path[y]);       //reading the image
		size_count++;
		

		cv::resize(input_img, input_img, cv::Size(64, 128));   //resizing (dimension used is that suggested from paper)
		d1.compute(input_img, Roi_descriptors, Size(0, 0), Size(0, 0), Roi_locations); //HOG computation

		for (int i = 0; i < Roi_descriptors.size(); i++) {
			HOGFeat_train.at<float>(size_count - 1, i) = Roi_descriptors.at(i); //saving the extracted descrptors as a row in a Mat object
		}

		labels_train.push_back(1);   // saving +1 if the image was considered as a positive sample

	}
	
	cout << "number of positive sample for training: " << size_count << std::endl;

/********************************************************************************************************************************/
	//variable initialization for the sliding window approach
	int windows_n_rows = 256;
	int windows_n_cols = 256;
	int StepSlide = 128;

	int window_count = 0;
	int negative = 0;

	for (int y = 0; y < negative_image_path.size(); y++) {

		input_img = imread(negative_image_path[y]);   
		
		// reading the image
		Mat DrawResultGrid = input_img.clone();
		/************************* starting the sliding window approach      **************************/

		for (int row = 0; row <= input_img.rows - windows_n_rows; row += StepSlide)
		{
			for (int col = 0; col <= input_img.cols - windows_n_cols; col += StepSlide)
			{
				Rect windows(col, row, windows_n_rows, windows_n_cols);
				Mat Roi1 = input_img(windows);
			
				Mat DrawResultHere = input_img.clone();
				
				
				rectangle(DrawResultHere, windows, Scalar(0, 255, 255), 1, 8, 0);
				
						// Draw only rectangle
						// Draw grid
				rectangle(DrawResultGrid, windows, Scalar(0, 255, 255), 1, 8, 0);

				/*			// Show  rectangle
				namedWindow("Step 2 draw Rectangle", WINDOW_AUTOSIZE);
				resize(DrawResultHere, DrawResultHere, cv::Size(500, 200));
				imshow("Step 2 draw Rectangle", DrawResultHere);
				waitKey(100);
				imwrite("Step2.JPG", DrawResultHere);*/


				/*		// Show grid
				namedWindow("Step 3 Show Grid", WINDOW_AUTOSIZE);
				
				imshow("Step 3 Show Grid", DrawResultGrid);
				waitKey(100);
				imwrite("Step3.JPG", DrawResultGrid);*/


						


							// Select windows roi
				

				/*		//Show ROI
				namedWindow("Step 4 Draw selected Roi", WINDOW_AUTOSIZE);
				imshow("Step 4 Draw selected Roi", Roi1);
				
				imwrite("Step4.JPG", Roi1);
				waitKey(0);*/
				size_count++;
				// rectangle coordinates defined by sliding window approach 
				
				negative++;

				// image portion associated to sliding window approach
				Mat Roi = input_img(windows);

				resize(Roi, Roi, cv::Size(64, 128));     // computation of HOG (same step as above)
				d1.compute(Roi, Roi_descriptors, Size(0, 0), Size(0, 0), Roi_locations);

				for (int i = 0; i < Roi_descriptors.size(); i++) {
					HOGFeat_train.at<float>(size_count - 1, i) = Roi_descriptors.at(i); //saving features extracted
				}

				labels_train.push_back(-1);       //the labels is -1 when considering negative samples
			}

		}
	}

	
	std::cout << "negative sample: " << negative << "\n" << std::endl;
	std::cout << "labels train positive + negative: " << labels_train.size() << "\n" << std::endl;
	std::cout << "starting to train a POLY svm..." << std::endl;

/********************************************************************************************************************************/
	//SVM parameter initialization
	Ptr<SVM> svm = SVM::create();      // the svm is commented so as to not block the program execution for long time
	/*svm->setC(0.01);
	svm->setCoef0(0.1);
	svm->setDegree(4);
	svm->setGamma(0.3);
	svm->setCoef0(0);
	svm->setKernel(SVM::POLY);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 100000, 1e-6));
	svm->setType(SVM::C_SVC);
	//svm training
	svm->train(HOGFeat_train, ROW_SAMPLE, labels_train);*/

	//saving svm
	svm->save("filtering_svm_poly.xml");


	return 0;
}


