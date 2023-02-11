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


using namespace std;
using namespace cv;
using namespace ml;


double intersection_prediction(Rect ground_truth, Rect prediction);
/**********************************************************************************************************/
/*  input = two rectangles (in principle one representing the ground truth, the other the prediction)
	output = Intersection of Union between the two rectangles
																		 */
/**********************************************************************************************************/


double findBestIou(std::vector<std::vector<double>> IoU_vect);
/**********************************************************************************************************/
/*  input = Intersection of union of prediction of rectangles for a image
	output = best value between IoU wrt all rectangles considered
																		 */
 /**********************************************************************************************************/


double findAverageIou(std::vector<std::vector<double>> IoU_vect);
/**********************************************************************************************************/
/*  input = Intersection of union of prediction of rectangles for a image
	output = average value between IoU wrt all rectangles considered
																		 */
/**********************************************************************************************************/


Mat filtering(std::vector<Rect> kaggle_obj_dect, Mat input_img, std::vector<Rect> coord, Ptr<SVM> svm, int i, string path);
/**********************************************************************************************************/
/*  input = this function take the considered image, the prediction and ground_truth associated to
*			the image and then evaluate the final performance after the filter of the obtained result
*			through a pretrained SVM
*
	output = image obtained after histogram equalization of single channell
																			 */
/**********************************************************************************************************/



std::vector<Rect> extractingRect(vector<cv::String> test_path_venice, cv::String test_labels_path, int i, bool p);
/**********************************************************************************************************/
/*  input = function that extrach the ground_truth information about the test image considered

	output = rectangles description done during the construction of the ground truth dataset
																								*/
/**********************************************************************************************************/


bool IoU(const cv::Rect& lhs, const cv::Rect& rhs, double th);
/**********************************************************************************************************/
/*  input = two rectangles

	output = bool operation that is true if the IoU between two input
	rectangles is above a certain threshold
																		*/
/**********************************************************************************************************/



int main(void) {


/*********************************************************************************************************************************/
	// loading the trained SVM on HOG features 
	cout << "downloading trained cascade of classifier and SVM " << "\n" << std::endl;
	
	Ptr<SVM> svm = SVM::create();
	svm = SVM::load("..\\SVM\\andreoli_jacopo_svm.xml");
	

/*********************************************************************************************************************************/
	// loading positive and negative test samples

	cv::String test_labels_path_venice = "..\\TEST_DATASET\\venice_labels_txt";
	vector<cv::String> test_path_venice;
	glob("..\\TEST_DATASET\\venice\\*.png", test_path_venice, false);


	cv::String test_labels_path_kaggle = "..\\TEST_DATASET\\kaggle_labels_txt";
	vector<cv::String> test_path_kaggle;
	glob("..\\TEST_DATASET\\kaggle\\*.jpg", test_path_kaggle, false);


/*********************************************************************************************************************************/
	// loading the trained cascade of classifier


	CascadeClassifier cascade_boat;
	string cascade_path = "..\\cascade\\cascade.xml";
	cascade_boat.load(cascade_path);


/*********************************************************************************************************************************/
	// variables initialization

	Mat test_image_venice, dest, test_image_kaggle;
	std::vector<Rect> rectangles_venice, labels_test_sample_venice, rectangles_kaggle, labels_test_sample_kaggle;


/*********************************************************************************************************************************/
	// param initialization
	cv::Size size = cv::Size(30, 30);
	int param = 10;


/*********************************************************************************************************************************/
	//main part of the program
/*********************************************************************************************************************************/
	cout << "start to detect images " << "\n" << std::endl;

	//analyzing the kaggle test dataset 
	for (int i = 0; i < test_path_kaggle.size(); i++) { // for each test image in dataset "kaggle"
		string path_to_result;
		path_to_result.append("..\\results\\result_kaggle_");
		path_to_result.append(to_string(i));
		path_to_result.append(".jpg");

		test_image_kaggle = imread(test_path_kaggle[i]);															// - reading the colored image (showing result)
		Mat test_image_kaggle_gray = imread(test_path_kaggle[i], CV_8UC1);											// - reading the bw image (object detection)

		labels_test_sample_kaggle = extractingRect(test_path_kaggle, test_labels_path_kaggle, i, 0);				// - ground_truth bounding box extraction
		cascade_boat.detectMultiScale(test_image_kaggle, rectangles_kaggle, 1.83, 1, 0);							// - cascade of classifier bound. box detection
		
		Mat result = filtering(rectangles_kaggle, test_image_kaggle, labels_test_sample_kaggle, svm, i, "Kaggle");  // - filtered result through binarization + SVM
		resize(result, result, cv::Size(800, 500));	
		cout << path_to_result << std::endl;
		imwrite(path_to_result, result);        // saving final result
		
		cvDestroyAllWindows();

		imshow("test image", result);
		waitKey(2500);
		labels_test_sample_kaggle.clear();

	}

/*********************************************************************************************************************************/
	cout << test_path_venice.size() << std::endl;
	//analyzing the venice test dataset 
	for (int i = 0; i < test_path_venice.size(); i++) {
		string path_to_result;
		path_to_result.append("..\\results\\result_venice_");
		path_to_result.append(to_string(i));
		path_to_result.append(".jpg");

		test_image_venice = imread(test_path_venice[i]);															//same procedure described above
		Mat test_image_venice_gray = imread(test_path_venice[i], CV_8UC1);                                    

		labels_test_sample_venice = extractingRect(test_path_venice, test_labels_path_venice, i, 1);
		cascade_boat.detectMultiScale(test_image_venice, rectangles_venice, 1.4, 10, 0, size);


		Mat result = filtering(rectangles_venice, test_image_venice, labels_test_sample_venice, svm, i, "Venice");
		resize(result, result, cv::Size(800, 500));
		imwrite(path_to_result, result);

		cvDestroyAllWindows();
		imshow("test image", result);
		imwrite(path_to_result, result);

		waitKey(2500);
		labels_test_sample_venice.clear();
	}

	return 0;

}














/****************************************************************************************************************/
std::vector<std::vector<cv::Rect>> cluster_rects_image(std::vector<cv::Rect> rects, double th)
{
	std::vector<int> labels;
	int n_labels = cv::partition(rects, labels, [th](const cv::Rect rect1, const cv::Rect rect2) {
		double i = double((rect1 & rect2).area());
		double intersection_rect1_area = i / static_cast<double>(rect1.area());
		double intersection_rect2_area = i / static_cast<double>(rect2.area());
		return (intersection_rect1_area > th) || (intersection_rect2_area > th);
	});

	std::vector<std::vector<cv::Rect>> clusters(n_labels);
	for (size_t i = 0; i < rects.size(); ++i) {
		clusters[labels[i]].push_back(rects[i]);
	}

	return clusters;
};
/****************************************************************************************************************/



/****************************************************************************************************************/
cv::Rect union_of_rects(std::vector<cv::Rect> cluster)
{
	cv::Rect union_rect;
	if (!cluster.empty())
	{
		union_rect = cluster[0];

		for (int i = 1; i < cluster.size(); i++) {
			union_rect |= cluster[i];
		}
	}
	return union_rect;
};
/****************************************************************************************************************/



/****************************************************************************************************************/
std::vector<int> extractValues(std::string s, char delim, char init_del)
{
	std::string portion = s.substr(s.find(init_del) + 1, s.size());
	std::vector<int> v;
	std::string i;
	std::stringstream ss(portion);
	while (std::getline(ss, i, delim))
	{
		std::stringstream c(i);
		int x;
		c >> x;
		v.push_back(x);
	}
	return v;
};
/****************************************************************************************************************/



/****************************************************************************************************************/
cv::Rect recoverRect(std::vector<int> coord) {

	//coords[0] coords[2] define the point of the top left corner
	//coords[1] coords[3] define the point of the bottom right corner

	Point topLeft, bottomRight;
	topLeft.x = coord[0];
	topLeft.y = coord[2];
	bottomRight.x = coord[1];
	bottomRight.y = coord[3];
	int width = abs(topLeft.x - bottomRight.x);
	int height = abs(topLeft.y - bottomRight.y);
	Rect rect = Rect(coord[0], coord[2], width, height);
	return rect;
};
/****************************************************************************************************************/



/****************************************************************************************************************/
double intersection_prediction(Rect ground_truth, Rect prediction) {
	double IoU;
	double intersection_area = (double)((ground_truth & prediction).area());
	IoU = (intersection_area / (double(ground_truth.area()) + double(prediction.area()) - intersection_area));
	return IoU;
};
/****************************************************************************************************************/



/****************************************************************************************************************/
double findBestIou(std::vector<std::vector<double>> IoU_vect) {
	double best = 0;
	std::vector<double> max_for_rect;
	for (int j = 0; j < IoU_vect.size(); j++) {
		double max = *max_element(IoU_vect[j].begin(), IoU_vect[j].end());
		max_for_rect.push_back(max);
	}
	best = *max_element(max_for_rect.begin(), max_for_rect.end());
	return best;
};


/****************************************************************************************************************/
double findAverageIou(std::vector<std::vector<double>> IoU_vect) {
	double average = 0;
	std::vector<double> average_for_rect;

	for (int j = 0; j < IoU_vect.size(); j++) {
		for (int i = 0; i < IoU_vect[j].size(); i++) {
			if (IoU_vect[j][i] != 0) {
				average_for_rect.push_back(IoU_vect[j][i]);
			}
		}

	}

	for (int y = 0; y < average_for_rect.size(); y++) {
		average += average_for_rect[y];
	}

	return (average / average_for_rect.size());

};
/****************************************************************************************************************/



/****************************************************************************************************************/
Mat filtering(std::vector<Rect> kaggle_obj_dect, Mat input_img, std::vector<Rect> coord, Ptr<SVM> svm, int i, string path) {

	Mat roi, dest, roi_otsu;
	Mat otsu_image;

	cvtColor(input_img, dest, CV_BGR2GRAY, 1);
	threshold(dest, otsu_image, 200, 255, THRESH_TOZERO);

	vector<Point> Roi_locations, BW_locations;

	vector<float> Roi_descriptors, BW_descriptors;
	HOGDescriptor d2(Size(64, 128), Size(16, 16), Size(8, 8), Size(8, 8), 9, 1);



	std::vector<cv::Rect> good_rect;

	for (int i = 0; i < kaggle_obj_dect.size(); i++) {

		roi = dest(kaggle_obj_dect[i]);
		roi_otsu = otsu_image(kaggle_obj_dect[i]);
		float countWhite = float(countNonZero(roi_otsu));
		float total_pix = roi.cols * roi.rows;
		float ratio = countWhite / total_pix;

		resize(roi, roi, cv::Size(64, 128));


		d2.compute(roi, Roi_descriptors, Size(0, 0), Size(0, 0), Roi_locations);
		if (svm->predict(Roi_descriptors) && ratio > 0.04) {
			cv::Rect one_good_rect = Rect(kaggle_obj_dect[i]);
			good_rect.push_back(one_good_rect);
		}
	}
	int false_positive = 0;
	std::vector<double> IoU;
	std::vector<std::vector<double>> IoU_image;
	std::vector<std::vector<cv::Rect>> clusters_of_image = cluster_rects_image(good_rect, 0.1);
	double count_clusters_rect = 0;
	for (int y = 0; y < clusters_of_image.size(); y++) {
		cv::Rect union_of_clust_rect = union_of_rects(clusters_of_image[y]);
		cv::rectangle(input_img, union_of_clust_rect, cv::Scalar(0, 255, 255), 2);
		count_clusters_rect++;
		for (int u = 0; u < coord.size(); u++) {

			IoU.push_back(intersection_prediction(coord[u], union_of_clust_rect));

		}

		std::vector<double> zero_vector(IoU.size(), 0.0);
		if (zero_vector == IoU) {
			false_positive++;
		}

		IoU_image.push_back(IoU);
		IoU.clear();
	}


	for (int j = 0; j < coord.size(); j++) {

		roi = input_img(coord[j]);

		cvtColor(input_img, dest, CV_BGR2GRAY, 1);

		cv::rectangle(input_img, coord[j], cv::Scalar(0, 255, 0), 2);
	}

	double BestIoU = findBestIou(IoU_image);
	double averageIoU = findAverageIou(IoU_image);

	cout << "intersection of union obtained for test " << path << ", image " << i << ": \n" << std::endl;
	cout << "- best Iou: " << BestIoU << std::endl;
	cout << "- average Iou: " << averageIoU  << std::endl;
	cout << "- n false positives: " << false_positive << " \n" << std::endl;


	return input_img;

};
/****************************************************************************************************************/



/****************************************************************************************************************/
std::vector<Rect> extractingRect(vector<cv::String> test_path_venice, cv::String test_labels_path, int i, bool p) {

	std::vector<Rect> labels_test_sample_venice;
	std::size_t pos;
	string image_path_venice = test_path_venice[i];
	if (p) {
		pos = image_path_venice.find("venice\\");
	}
	else {
		pos = image_path_venice.find("kaggle\\");
	}

	std::string imagenumber1 = image_path_venice.substr(pos + 7, 2);


	ifstream One_label;
	string label_path = test_labels_path;
	label_path.append("\\");
	label_path.append(imagenumber1);
	label_path.append(".txt");
	One_label.open(label_path);

	if (!One_label) {
		cout << "Unable to open .txt file ";
		cout << imagenumber1 << endl;

	}

	string line;	// one line of the document

	while (std::getline(One_label, line))
	{
		if (line.find("boat:") != string::npos)	// boat found!
		{
			std::vector<int> coords = extractValues(line, ';', ':');
			Rect rect = recoverRect(coords);
			labels_test_sample_venice.push_back(rect);
		}

		else {

			One_label.close();
		}
	}

	return labels_test_sample_venice;

};
/****************************************************************************************************************/


/****************************************************************************************************************/
bool IoU(const cv::Rect& lhs, const cv::Rect& rhs, double th) {
	double i = static_cast<double>((lhs & rhs).area());
	double ratio_intersection_over_lhs_area = i / static_cast<double>(lhs.area());
	double ratio_intersection_over_rhs_area = i / static_cast<double>(rhs.area());
	return (ratio_intersection_over_lhs_area > th) || (ratio_intersection_over_rhs_area > th);
};
/****************************************************************************************************************/

