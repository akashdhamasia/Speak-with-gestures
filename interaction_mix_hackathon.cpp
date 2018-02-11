#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/ml/ml.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include <vector>

#include <dlib/opencv/cv_image.h>
#include <dlib/opencv/to_open_cv.h>

#include <fstream>
#include <string>

#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sstream>

#include <time.h>

#include <unistd.h>

using namespace dlib;
using namespace std;
using namespace cv;
using namespace cv::ml;

// ----------------------------------------------------------------------------------------

void get_svm_detector(const Ptr<SVM>& svm, std::vector< float > & hog_detector )
{
    // get the support vectors
    Mat sv = svm->getSupportVectors();
    const int sv_total = sv.rows;
    // get the decision function
    Mat alpha, svidx;
    double rho = svm->getDecisionFunction(0, alpha, svidx);

    CV_Assert( alpha.total() == 1 && svidx.total() == 1 && sv_total == 1 );
    CV_Assert( (alpha.type() == CV_64F && alpha.at<double>(0) == 1.) ||
               (alpha.type() == CV_32F && alpha.at<float>(0) == 1.f) );
    CV_Assert( sv.type() == CV_32F );
    hog_detector.clear();

    hog_detector.resize(sv.cols + 1);
    memcpy(&hog_detector[0], sv.ptr(), sv.cols*sizeof(hog_detector[0]));
    hog_detector[sv.cols] = (float)-rho;
}

void draw_locations( Mat & img, const std::vector< Rect > & locations, const Scalar & color )
{
    if( !locations.empty() )
    {
        std::vector< Rect >::const_iterator loc = locations.begin();
        std::vector< Rect >::const_iterator end = locations.end();
        for( ; loc != end ; ++loc )
        {
            cv::rectangle( img, *loc, color, 2 );
        }
    }
}


int main(int argc, char** argv)
{  
    try
    {

        frontal_face_detector detector = get_frontal_face_detector();
        shape_predictor sp;
        deserialize(argv[1]) >> sp;

	point temp_point_max, temp_point_min;
	temp_point_max.x() = 0;
	temp_point_max.y() = 0;
	temp_point_min.x() = 1000;
	temp_point_min.y() = 1000;

	const char* new_name1;
	ostringstream convert1;
	int counter = 1;
	const char* new_name2;
	ostringstream convert2;
	const char* new_name3;
	ostringstream convert3;
	const char* new_name4;
	ostringstream convert4;
	const char* new_name5;
	ostringstream convert5;

	int displacement_x = 0;
	std::string gesture;

	int prev_dir = 0;
	int positive_dir = 0;
	int transaction_pos = 0;
	int transaction_neg = 0;

	int displacement_y = 0;
	std::string gesture_y;

	int prev_dir_y = 0;
	int positive_dir_y = 0;
	int transaction_pos_y = 0;
	int transaction_neg_y = 0;

	cv::Point current_position, prev_position;

	current_position.x = 0;
	current_position.y = 0;

	prev_position.x = 0;
	prev_position.y = 0;

	const Size size(64,128);// size(32,64);
	char key = 27;
	Scalar reference( 0, 255, 0 );
	Scalar trained( 0, 0, 255 );
	//Scalar trained( 255, 0, 255 );
	Mat img, draw;
	Ptr<SVM> svm;
	//HOGDescriptor hog;
	HOGDescriptor my_hog;
	my_hog.winSize = size;
	VideoCapture video;
	std::vector< Rect > locations;

	// Load the trained SVM.
	svm = StatModel::load<SVM>( "my_people_detector25.yml" );
	// Set the trained svm to my_hog
	std::vector< float > hog_detector;
	get_svm_detector( svm, hog_detector );
	my_hog.setSVMDetector( hog_detector );

	int counter_thumb = 0;
	int counter_no = 0;
	int counter_yes = 0;
	int counter_smile = 0;

	int question_no = 1;

        cv::VideoCapture cap(0);
        if (!cap.isOpened())
        {
            cerr << "Unable to connect to camera" << endl;
            return 1;
        }


        image_window win;
	cv::Mat mat_image;

	std::string result_survey[5];

	while(1)
        {

            cv::Mat org_frame;
            cap >> org_frame;
            cv_image<bgr_pixel> img(org_frame);

            std::vector<dlib::rectangle> dets = detector(img);
            //cout << "Number of faces detected: " << dets.size() << endl;

            std::vector<full_object_detection> shapes;
            for (unsigned long j = 0; j < dets.size(); ++j)
            {
		full_object_detection shape = sp(img, dets[j]);

		mat_image = org_frame.clone();

		Point stable_point;

		stable_point.x = shape.part(28).x();
		stable_point.y = shape.part(28).y();

		circle(mat_image, stable_point, 3, Scalar(255, 0, 0), 2, CV_AA);

		current_position = stable_point;

		displacement_x = current_position.x - prev_position.x;
		displacement_y = current_position.y - prev_position.y;

		prev_position = stable_point;

		if(displacement_x > 5)
		{
		positive_dir = 1;

		if(prev_dir != positive_dir)
		{		
			if(transaction_pos == 1){
			gesture = "no";
			transaction_pos = 0;
			transaction_neg = 0;
			transaction_neg_y = 0;
			transaction_pos_y = 0;
			}
			else{
			transaction_pos = 1;
			}
		}	


		}
		else if(displacement_x < -5)
		{
		positive_dir = 0;

		if(prev_dir != positive_dir)
		{		
			if(transaction_neg == 1){
			gesture = "no";
			transaction_neg = 0;
			transaction_pos = 0;
			transaction_neg_y = 0;
			transaction_pos_y = 0;
			}
			else{
			transaction_neg = 1;
			}
		}

		}
		else
		{
		gesture = "unknown";
		//positive_dir = -1;
		}

		if(displacement_y > 5)
		{
		positive_dir_y = 1;

		if(prev_dir_y != positive_dir_y)
		{		
			if(transaction_pos_y == 1){
			gesture_y = "yes";
			transaction_pos_y = 0;
			transaction_neg_y = 0;
			transaction_pos = 0;
			transaction_neg = 0;
			}
			else{
			transaction_pos_y = 1;
			}
		}	


		}
		else if(displacement_y < -5)
		{
		positive_dir_y = 0;

		if(prev_dir_y != positive_dir_y)
		{		
			if(transaction_neg_y == 1){
			gesture_y = "yes";
			transaction_neg_y = 0;
			transaction_pos_y = 0;
			transaction_pos = 0;
			transaction_neg = 0;
			}
			else{
			transaction_neg_y = 1;
			}
		}

		}
		else
		{
		gesture_y = "unknown";
		//positive_dir = -1;
		}


		prev_dir = positive_dir;
		prev_dir_y = positive_dir_y;

		//if(gesture == "no" || gesture_y == "yes")	
		//std::cout << gesture << "   " << gesture_y << std::endl;

///////////////////////for smile//////////////////////////////////////////////////

		//lips part

		cv::Point lip_point_min;
		cv::Point lip_point_max;

		Point lip_left;
		Point lip_right;

		lip_point_min.x = 1000;
		lip_point_min.y = 1000;
		lip_point_max.x = 0;
		lip_point_max.y = 0;

		for(int i=49; i<=60; i++)
		{
			if(lip_point_max.x < shape.part(i).x()){
			lip_point_max.x = (int)shape.part(i).x();
			lip_right.x = (int)shape.part(i).x();
			lip_right.y = (int)shape.part(i).y();
			}

			if(lip_point_max.y < shape.part(i).y())
			lip_point_max.y = (int)shape.part(i).y();

			if(lip_point_min.x > shape.part(i).x()){
			lip_point_min.x = (int)shape.part(i).x();
			lip_left.x = (int)shape.part(i).x();
			lip_left.y = (int)shape.part(i).y();
			}

			if(lip_point_min.y > shape.part(i).y())
			lip_point_min.y = (int)shape.part(i).y();

		}

		circle(mat_image, lip_left, 3, Scalar(255, 0, 0), 2, CV_AA);
		circle(mat_image, lip_right, 3, Scalar(255, 0, 0), 2, CV_AA);	

		//eye part
		cv::Point eye_left_min;
		cv::Point eye_left_max;

		Point eye_left;
		Point eye_right;

		eye_left_min.x = 1000;
		eye_left_min.y = 1000;
		eye_left_max.x = 0;
		eye_left_max.y = 0;

		for(int i=36; i<=41; i++)
		{
			if(eye_left_max.x < shape.part(i).x())
			eye_left_max.x = (int)shape.part(i).x();

			if(eye_left_max.y < shape.part(i).y())
			eye_left_max.y = (int)shape.part(i).y();

			if(eye_left_min.x > shape.part(i).x()){
			eye_left_min.x = (int)shape.part(i).x();
			eye_left.x = (int)shape.part(i).x();
			eye_left.y = (int)shape.part(i).y();
			}

			if(eye_left_min.y > shape.part(i).y())
			eye_left_min.y = (int)shape.part(i).y();

		}

		cv::Point eye_right_min;
		cv::Point eye_right_max;

		eye_right_min.x = 1000;
		eye_right_min.y = 1000;
		eye_right_max.x = 0;
		eye_right_max.y = 0;

		for(int i=42; i<=47; i++)
		{
			if(eye_right_max.x < shape.part(i).x()){
			eye_right_max.x = (int)shape.part(i).x();
			eye_right.x = (int)shape.part(i).x();
			eye_right.y = (int)shape.part(i).y();
			}

			if(eye_right_max.y < shape.part(i).y())
			eye_right_max.y = (int)shape.part(i).y();

			if(eye_right_min.x > shape.part(i).x())
			eye_right_min.x = (int)shape.part(i).x();

			if(eye_right_min.y > shape.part(i).y())
			eye_right_min.y = (int)shape.part(i).y();

		}

		circle(mat_image, eye_left, 3, Scalar(255, 0, 0), 2, CV_AA);
		circle(mat_image, eye_right, 3, Scalar(255, 0, 0), 2, CV_AA);

		int eye_diff = eye_right.x - eye_left.x;
		int lip_diff = lip_right.x - lip_left.x;

		double smile_detected = (float)eye_diff/lip_diff;

		//std::cout << "smile_detected  " << smile_detected << std::endl; 

		//if(smile_detected < 1.6) 
		//std::cout << "smile_detected  " << std::endl; 

//////////////////////////////////smile end//////////////////////////////////////////////////

		locations.clear();
		//my_hog.detectMultiScale( img, locations );
		my_hog.detectMultiScale(mat_image, locations, 0, Size(8,8), Size(32,32), 1.09, 2);
		draw_locations( mat_image, locations, trained );

////////////////////////////GESTURE DECISION////////////////////////////////////////////////////

if(locations.size() > 0 && question_no == 5 )
{
	counter_thumb++;
	counter_no = 0;
	counter_yes = 0;
	counter_smile = 0;
}
else if(gesture == "no" && (question_no == 1 || question_no == 2 || question_no == 3))
{
	counter_no++;
	counter_thumb = 0;
	counter_yes = 0;
	counter_smile = 0;
}
else if(gesture_y == "yes" && (question_no == 1 || question_no == 2 || question_no == 3))
{
	counter_yes++;
	counter_thumb = 0;
	counter_no = 0;
	counter_smile = 0;
}
else if(smile_detected < 1.6 && question_no == 4)
{
	counter_smile++;
	counter_thumb = 0;
	counter_no = 0;
	counter_yes = 0;
}
else
{
	counter_smile = 0;
	counter_thumb = 0;
	counter_no = 0;
	counter_yes = 0;
}

convert1 << "honeywell/Q" << question_no << ".png";

//std::cout << convert1 << std::endl;

//new_name1 = convert1.str().c_str();

Mat container = Mat(480, 1280, mat_image.type(), Scalar(255,255,255)); //change
Mat container_sidebar = imread(convert1.str().c_str());
resize(container_sidebar, container_sidebar, cv::Size(640, 480));

Mat layout1_container = Mat(container, cv::Rect(0, 0, 640, 480));
Mat layout2_container = Mat(container, cv::Rect(640, 0, 640, 480));

mat_image.copyTo(layout1_container);
container_sidebar.copyTo(layout2_container);


if(counter_thumb > 6 && question_no == 5 )
{
std::cout << " LIKE DETECTED " << std::endl;

result_survey[question_no-1] = "like";

counter_thumb = 0;
counter_no = 0;
counter_yes = 0;
counter_smile = 0;

putText(container, "'LIKE' DETECTED", cvPoint(880,420), FONT_HERSHEY_PLAIN, 2.5, Scalar(255, 0, 0), 2.5, 8);
cv::imshow("mat_image", container);
cv::waitKey(1000);

question_no++;

if(question_no > 6)
return 1;

}
else if(counter_no > 6 && (question_no == 1 || question_no == 2 || question_no == 3))
{
std::cout << " NO DETECTED " << std::endl;

result_survey[question_no-1] = "no";

counter_thumb = 0;
counter_no = 0;
counter_yes = 0;
counter_smile = 0;

putText(container, "'NO' DETECTED", cvPoint(880,420), FONT_HERSHEY_PLAIN, 2.5, Scalar(255, 0, 0), 2.5, 8);
cv::imshow("mat_image", container);
cv::waitKey(1000);

question_no++;

if(question_no > 6)
return 1;

}
else if(counter_yes > 6 && (question_no == 1 || question_no == 2 || question_no == 3))
{
std::cout << " YES DETECTED " << std::endl;

result_survey[question_no-1] = "yes";

counter_thumb = 0;
counter_no = 0;
counter_yes = 0;
counter_smile = 0;

putText(container, "'YES' DETECTED", cvPoint(880,420), FONT_HERSHEY_PLAIN, 2.5, Scalar(255, 0, 0), 2.5, 8);
cv::imshow("mat_image", container);
cv::waitKey(1000);

question_no++;

if(question_no > 6)
return 1;

}
else if(counter_smile > 6 && question_no == 4)
{
std::cout << " SMILE DETECTED " << std::endl;

result_survey[question_no-1] = "smile";

counter_thumb = 0;
counter_no = 0;
counter_yes = 0;
counter_smile = 0;

putText(container, "'SMILE' DETECTED", cvPoint(880,420), FONT_HERSHEY_PLAIN, 2.5, Scalar(255, 0, 0), 2.5, 8);
cv::imshow("mat_image", container);
cv::waitKey(1000);

question_no++;

if(question_no > 6)
return 1;

}
else if(question_no == 6)
{

counter_thumb = 0;
counter_no = 0;
counter_yes = 0;
counter_smile = 0;

putText(container, result_survey[0], cvPoint(780,110), FONT_HERSHEY_PLAIN, 2, Scalar(255, 0, 0), 2, 8);
putText(container, result_survey[1], cvPoint(780,200), FONT_HERSHEY_PLAIN, 2, Scalar(255, 0, 0), 2, 8);
putText(container, result_survey[2], cvPoint(780,285), FONT_HERSHEY_PLAIN, 2, Scalar(255, 0, 0), 2, 8);
putText(container, result_survey[3], cvPoint(780,390), FONT_HERSHEY_PLAIN, 2, Scalar(255, 0, 0), 2, 8);
putText(container, result_survey[4], cvPoint(780,460), FONT_HERSHEY_PLAIN, 2, Scalar(255, 0, 0), 2, 8);

//cv::imshow("mat_image", container);
//cv::waitKey(1000);

}


		cv::imshow("mat_image", container);
		cv::waitKey(1);	

		shapes.push_back(shape);

	        convert1.str("");
	        convert1.clear();
            }

            // Now lets view our face poses on the screen.
            //win.clear_overlay();
            //win.set_image(img);
            //win.add_overlay(render_face_detections(shapes));

        }

    }
    catch (exception& e)
    {
        cout << "\nexception thrown!" << endl;
        cout << e.what() << endl;
    }
}

// ----------------------------------------------------------------------------------------

