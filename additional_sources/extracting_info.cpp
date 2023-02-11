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


std::vector<int> extractValues(std::string s, char delim, char init_del);
/**********************************************************************************************************/
/*  input = line of the ground_truth document (the info are assumed to be setted as standard adopted)
	output = coordinate of the upper left and bottom right rectangle defined during
			 the construction of the ground_truth dataset (further description in the report)
																								 */
/**********************************************************************************************************/


																		
cv::Rect recoverRect(std::vector<int> coord);
/**********************************************************************************************************/
/*  input = coordinates of upper left & bottom right rectangle description
	output = rectangle obtained through input description
																		 */
/**********************************************************************************************************/




std::vector<Rect> extractingRect(vector<cv::String> test_path_venice, cv::String test_labels_path, int i, bool p);
/**********************************************************************************************************/
/*  input = function that extrach the ground_truth information about the test image considered

	output = rectangles description done during the construction of the ground truth dataset
																								*/


std::vector<int> recoverCoords(std::vector<int> coord);
/**********************************************************************************************************/
/*  input = coordinates of upper left & bottom right exttrema of rectangles

	output = rewriting  the rectangle specification in term of (x, y, width, height)
																						*/
/**********************************************************************************************************/


void writeNegativeLabels(int i, string path_to_file);
/**********************************************************************************************************/
/*  input = absolute path where save extracted negative sample from ground truth

	output = .txt file description of the negative samples
																						*/
/**********************************************************************************************************/


void writePositiveLabels(std::vector<std::vector<int>> labels, string imagenumber, string path_to_file);
/**********************************************************************************************************/
/*  input = absolute path where save extracted positive sample from ground truth

	output = .txt file description of the positive sample
																						*/
/**********************************************************************************************************/

int main(void) {

/**********************************************************************************************************************/
	// variable initialization

	vector<cv::String> labels_positive;
	vector<cv::String> negative_path;
	string images_train;
	std::vector<std::vector<int>> labels_one_sample;
	String Training_labels_Path;
/**********************************************************************************************************************/

	//setting directory to folder for image processing

	glob("C:\\Users\\jacop_\\Desktop\\computer_vision_final_project\\NEGATIVE_SAMPLE\\*.jpg", negative_path, false);
	glob("C:\\Users\\jacop_\\Desktop\\FINAL_DATASET\\TRAINING_DATASET\\LABELS_TXT\\*.txt", labels_positive, false);
	images_train= "C:\\Users\\jacop_\\Desktop\\FINAL_DATASET\\TRAINING_DATASET\\IMAGES";
	Training_labels_Path = "C:\\Users\\jacop_\\Desktop\\FINAL_DATASET\\TRAINING_DATASET\\LABELS_TXT";

/**********************************************************************************************************************/

	cout << "there are: " << negative_path.size() << " negative samples" << std::endl;
	cout << "there are: " << labels_positive.size() << " positive samples" << std::endl;

	for (int i = 0; i < labels_positive.size(); i++) { // for each detected image

		ifstream sample;

		string image_path = labels_positive[i];
		std::size_t pos = image_path.find("LABELS_TXT\\image");
		std::string imagenumber1 = image_path.substr(pos + 16, 4);    //recovering the number of the image
																	  // we must ensure that it is matched with the correct labels

		string image_path_prova = images_train;
		image_path_prova.append("\\image");
		image_path_prova.append(imagenumber1);
		image_path_prova.append(".png");

		string label_path = Training_labels_Path;
		label_path.append("\\image");
		label_path.append(imagenumber1);
		label_path.append(".txt");
		sample.open(label_path);

		if (!sample) {
			continue;
		}

/********************************************************************************************************************************/
		// parsing the ground truth .txt file - training positive sample with relative labels

		string line;

		while (std::getline(sample, line))									    
		{
			if (line.find("boat:") != string::npos)	                            // we are considering both boat and hiddenboat
			{
	
				std::vector<int> coords = extractValues(line, ';', ':');
				std::vector<int> new_coords = recoverCoords(coords);
				labels_one_sample.push_back(new_coords);
				
				Mat input = imread(image_path_prova);  // the image is not cropped! 
				string path = "C:\\Users\\jacop_\\Desktop\\computer_vision_final_project\\final_project_cascade\\pos\\img";
				path.append(imagenumber1);
				path.append(".jpg");
				cv::imwrite(path, input);			   // saving image in the selected path 

			}
			else {
				sample.close(); // close if the line are finished
			}

		}

		writePositiveLabels(labels_one_sample, imagenumber1, "C:\\Users\\jacop_\\Desktop\\computer_vision_final_project\\final_project_cascade\\info.txt");
		cout << "saving positive sample: " << imagenumber1 << std::endl;
		labels_one_sample.clear();
		
	}

/********************************************************************************************************************************/
		// parsing the negative images extracted from both kaggle and venice dataset

	for (int i = 0; i < negative_path.size(); i++) {

		string index = to_string(i);
		string image_path = negative_path[i];
		std::string imagenumber = index; 	// retrieve image number
		Mat temp = imread(image_path);

		string path = "C:\\Users\\jacop_\\Desktop\\computer_vision_final_project\\final_project_cascade\\neg\\img";
		path.append(imagenumber);
		path.append(".jpg");
		cv::imwrite(path, temp);
		writeNegativeLabels(i, "C:\\Users\\jacop_\\Desktop\\computer_vision_final_project\\final_project_cascade\\bg.txt");
		cout << "saving: negative sample " << imagenumber << std::endl;
	}
	return 0;
	
}







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
std::vector<int> recoverCoords(std::vector<int> coord) {
	//coords[0] coords[2] define the point of the top left corner
	//coords[1] coords[3] define the point of the bottom right corner 

	Point topLeft, bottomRight;
	topLeft.x = coord[0];
	topLeft.y = coord[2];
	bottomRight.x = coord[1];
	bottomRight.y = coord[3];
	int width = abs(topLeft.x - bottomRight.x);
	int height = abs(topLeft.y - bottomRight.y);
	std::vector<int> rect = { coord[0], coord[2], width, height };
	return rect;
};
/****************************************************************************************************************/



/****************************************************************************************************************/
void writeNegativeLabels(int i, string path_to_file)
{
	ofstream outputfile;
	string index = to_string(i);
	outputfile.open(path_to_file, fstream::app);
	string path = "neg/img";
	path.append(index);
	path.append(".jpg");
	outputfile << path << std::endl;
};
/****************************************************************************************************************/



/****************************************************************************************************************/
void writePositiveLabels(std::vector<std::vector<int>> labels, string imagenumber, string path_to_file)
{
	ofstream outputfile;
	outputfile.open(path_to_file, fstream::app);
	string path = "pos/img";
	path.append(imagenumber);
	path.append(".jpg");

	string result;
	for (int i = 0; i < labels.size(); i++) {

		result.append(to_string(labels[i][0]));
		result.append(" ");
		result.append(to_string(labels[i][1]));
		result.append(" ");
		result.append(to_string(labels[i][2]));
		result.append(" ");
		result.append(to_string(labels[i][3]));
		result.append("  ");
	}

	outputfile << path << " " << labels.size() << " " << result << std::endl;
};
/****************************************************************************************************************/

