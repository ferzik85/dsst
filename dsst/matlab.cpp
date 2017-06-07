//#include "DSSTTracker.h"
//#include "MOSSETracker.h"
//#include <Windows.h>
//#include <limits>
//#include <cstddef>
//#include <fstream> // for matlab
//
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/highgui/highgui_c.h>
//#include <opencv/cv.h>
//
//
//using namespace std;
//using namespace cv;
//using namespace ctr;
//using namespace mosse;
//
//int keyboard;
//void processImagesDSST(int left, int top, int width, int height, int count, int nzeros, char* fistFrameFilename);

//
//LARGE_INTEGER frequency;        // ticks per second
//LARGE_INTEGER t1, t2;           // ticks
//double elapsedTime;
//
//int main(int argc, char* argv[])
//{
//	int left, top, width, height, count, trackerId, nzeros;
//	char* path;
//
//	if (argc == 9) {
//		trackerId = atoi(argv[1]);
//		left = atoi(argv[2]);
//		top = atoi(argv[3]);
//		width = atoi(argv[4]);
//		height = atoi(argv[5]);
//		count = atoi(argv[6]); // число картинок, которые будут прсомотрены последовательно
//		nzeros = atoi(argv[7]);
//		path = argv[8];
//		
//		cout << argv[0] << ' ' << trackerId << ' ' << left   << ' ' << top  << ' ' << width << ' '
//			 << height  << ' ' << count     << ' ' << nzeros << ' ' << path << endl;
//		processImagesDSST(left, top, width, height, count, nzeros, path);
//	}
//	else {
//		cout << "you skip proper command arguments" << endl;
//	}
//	
//	return 0;
//}
//
//void processImagesDSST(int left, int top, int width, int height, int count, int nzeros, char* fistFrameFilename) {
//
//	bool use_gray = false; // true - fast but maybe less precise, false - slower tracking but better
//	cv::Mat Im;
//	if (use_gray == true)
//		Im = imread(fistFrameFilename, CV_LOAD_IMAGE_GRAYSCALE);
//	else
//		Im = imread(fistFrameFilename);
//
//	if (Im.empty()) { cerr << "Unable to open first image frame: " << fistFrameFilename << endl; exit(EXIT_FAILURE); }
//	cv::Mat ImRGBRes; keyboard = 0;
//	dsst_tracker dsst;
//	bool init = false;
//	unsigned char *dataYorR; unsigned char *dataG; unsigned char *dataB;
//	string fn(fistFrameFilename);
//
//	ofstream rects; rects.open("D:\\rects.txt");
//	ofstream fpsfile; fpsfile.open("D:\\fps.txt");
//	int counter = 0;
//	double maxfps = 0.0;
//	double avgfps = 0.0;
//	double minfps = std::numeric_limits<float>::max();
//	int mag = 1;
//	while ((char)keyboard != 27)
//	{
//		counter++;
//		cv::Size dsize = cv::Size(Im.cols * mag, Im.rows * mag);
//		cv::resize(Im, ImRGBRes, dsize, 0, 0, INTER_LINEAR);
//		if (use_gray == true) {}
//		else
//			cv::cvtColor(Im, Im, CV_BGR2RGB);
//		//get the frame number and write it on the current frame
//		size_t index = fn.find_last_of("/");
//		if (index == string::npos) { index = fn.find_last_of("\\"); }
//		size_t index2 = fn.find_last_of("."); string prefix = fn.substr(0, index + 1); string suffix = fn.substr(index2);
//		string frameNumberString = fn.substr(index + 1, index2 - index - 1); istringstream iss(frameNumberString);
//		int frameNumber = 0; iss >> frameNumber;
//		cv::putText(ImRGBRes, "Frame: " + frameNumberString, Point(5, 15), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(29, 45, 255));
//		int w = Im.cols; int h = Im.rows;
//		if (init == false) { // if (frameNumber == 0) {
//			dataYorR = new unsigned char[Im.rows*Im.cols];
//			dataG = new unsigned char[Im.rows*Im.cols];
//			dataB = new unsigned char[Im.rows*Im.cols];
//		}
//
//		if (use_gray == true) {
//			for (int j = 0; j < Im.rows; j++)
//			{
//				uchar* p = Im.ptr<uchar>(j);
//				for (int i = 0; i < Im.cols; i++)
//				{
//					dataYorR[i + j*Im.cols] = p[i];
//					dataG[i + j*Im.cols] = p[i];
//					dataB[i + j*Im.cols] = p[i];
//				}
//			}
//		}
//		else
//			for (int j = 0; j < Im.rows; j++)
//			{
//				for (int i = 0; i < Im.cols; i++)
//				{
//					dataYorR[i + j*Im.cols] = Im.at<cv::Vec3b>(j, i)[0];
//					dataG[i + j*Im.cols] = Im.at<cv::Vec3b>(j, i)[1];
//					dataB[i + j*Im.cols] = Im.at<cv::Vec3b>(j, i)[2];
//				}
//			};
//
//		int cx = 0; int cy = 0; int rw = 0; int rh = 0; float score = 0;
//
//		if (init == false) {
//
//			//dsst.initializeTargetModel(x_center, y_center, obj_width, obj_height, w, h, dataYorR, dataG, dataB);  // example
//			//dsst.initializeTargetModel(200 + 25  - 1, 112 + 25 - 1, 50, 50, w, h, dataYorR, dataG, dataB);   // ball
//			//dsst.initializeTargetModel(189 + 21 - 1, 211 + 56 - 1, 42, 112, w, h, dataYorR, dataG, dataB);   // basketball
//			//dsst.initializeTargetModel(154 + 9 - 1, 94 + 24 - 1, 18, 48, w, h, dataYorR, dataG, dataB);   // bicycle
//			//dsst.initializeTargetModel(324 + 24 - 1, 162 + 30 - 1, 48, 60, w, h, dataYorR, dataG, dataB);   // bolt
//			//dsst.initializeTargetModel(6 + 22  - 1, 166 + 14 - 1, 43, 27, w, h, dataYorR, dataG, dataB);   // car
//			//dsst.initializeTargetModel(150 + 41  - 1, 56 + 48 - 1, 82, 96, w, h, dataYorR, dataG, dataB);   // david
//			//dsst.initializeTargetModel(175 + 19  - 1, 17 + 81 - 1, 37, 162, w, h, dataYorR, dataG, dataB);   // diving
//			//dsst.initializeTargetModel(385 + 53 - 1, 142 + 45 - 1, 105, 91, w, h, dataYorR, dataG, dataB);   // drunk
//			//dsst.initializeTargetModel(147 + 42  - 1, 114 + 109 - 1, 83, 217, w, h, dataYorR, dataG, dataB);   // fernando
//			//dsst.initializeTargetModel(126 + 24  - 1, 21 + 18 - 1, 47, 36, w, h, dataYorR, dataG, dataB);   // fish1
//			//dsst.initializeTargetModel(266 + 55  - 1, 118 + 40 - 1, 110, 80, w, h, dataYorR, dataG, dataB);   // fish2
//			//dsst.initializeTargetModel(136 + 21  - 1, 28 + 62 - 1, 42, 124, w, h, dataYorR, dataG, dataB);   // gymnastics
//			//dsst.initializeTargetModel(225 + 22  - 1, 92 + 23 - 1, 44, 46, w, h, dataYorR, dataG, dataB); // hand1	
//			//dsst.initializeTargetModel(150 + 23  - 1, 69 + 26 - 1, 47, 53, w, h, dataYorR, dataG, dataB);   // hand2		
//			//dsst.initializeTargetModel(111 + 13  - 1, 98 + 51 - 1, 26, 102, w, h, dataYorR, dataG, dataB);   // jogging
//			//dsst.initializeTargetModel(96 + 75 - 1, 58 + 73 - 1, 149, 145, w, h, dataYorR, dataG, dataB);   // motocross
//			//dsst.initializeTargetModel(439 + 24  - 1, 108 + 36 - 1, 49, 71, w, h, dataYorR, dataG, dataB);   // polarbear
//			//dsst.initializeTargetModel(161 + 18  - 1, 174 + 44 - 1, 36, 88, w, h, dataYorR, dataG, dataB);   // skating
//			//dsst.initializeTargetModel(140 + 44  - 1, 59 + 44 - 1, 88, 88, w, h, dataYorR, dataG, dataB);   // sphere
//			//dsst.initializeTargetModel(77 + 19 - 1, 93 + 25 - 1, 38, 51, w, h, dataYorR, dataG, dataB);   // sunshade
//			//dsst.initializeTargetModel(40 + 10  - 1, 64 + 23 - 1, 20, 47, w, h, dataYorR, dataG, dataB);   // surfing
//			//dsst.initializeTargetModel(168 + 25  - 1, 80 + 25 - 1, 50, 50, w, h, dataYorR, dataG, dataB);   // torus
//			//dsst.initializeTargetModel(145 + 34  - 1, 77 + 37 - 1, 68, 74, w, h, dataYorR, dataG, dataB);   // trellis
//			//dsst.initializeTargetModel(200 + 27  - 1, 330 + 44 - 1, 54, 88, w, h, dataYorR, dataG, dataB);   // tunnel
//			//dsst.initializeTargetModel(207 + 15  - 1, 117 + 51 - 1, 30, 102, w, h, dataYorR, dataG, dataB);   // woman
//		
//			if (use_gray == true)
//				dsst.initializeTargetModel(left + (int)floor(width / 2), top + (int)floor(height / 2), width, height, w, h, dataYorR);   // dog
//			else
//				dsst.initializeTargetModel(left + (int)floor(width / 2), top + (int)floor(height / 2), width, height, w, h, dataYorR, dataG, dataB);   // dog
//
//			//if (use_gray == true)
//			//	dsst.initializeTargetModel(139 + 25 - 1, 112 + 18 - 1, 51, 36, w, h, dataYorR);   // dog
//			//else
//			//	dsst.initializeTargetModel(139 + 25 - 1, 112 + 18 - 1, 51, 36, w, h, dataYorR, dataG, dataB);   // dog
//
//																												//dsst.initializeTargetModel(139 + 25 - 1, 112 + 18 - 1, 51, 36, w, h, dataYorR);   // dog
//																												//dsst.initializeTargetModel(280 + 44 - 1, 204 + 40 - 1, 88, 80, w, h, dataYorR, dataG, dataB);   // arma
//																												//dsst.initializeTargetModel(302 + 55 - 1, 156 + 25 - 1, 110, 50, w, h, dataYorR, dataG, dataB);   // tank1
//																												//dsst.initializeTargetModel(7 + 314 - 1, 217 + 129 - 1, 628, 258, w, h, dataYorR, dataG, dataB);   // tank2
//		}
//
//		double fps = 0;
//
//		if (init == true) {
//			QueryPerformanceFrequency(&frequency);
//			QueryPerformanceCounter(&t1);
//
//			if (use_gray == true)
//				dsst.findNextLocation(dataYorR);
//			else
//				dsst.findNextLocation(dataYorR, dataG, dataB);
//
//			QueryPerformanceCounter(&t2);
//			fps = double(frequency.QuadPart) / double((t2.QuadPart - t1.QuadPart));
//
//			if (minfps > fps) minfps = fps;
//			if (maxfps < fps) maxfps = fps;
//			avgfps += fps;
//		}
//
//		dsst.getNewLocationCoordinates(cx, cy, rw, rh, score);
//
//		init = true;
//		int  l = (cx - rw / 2); int  t = (cy - rh / 2);
//		int  r = (cx + rw / 2); int  b = (cy + rh / 2);
//		cv::rectangle(ImRGBRes, Point(mag * l, mag * t), Point(mag * r, mag * b), cvScalar(0, 255, 0), 1);
//
//		stringstream ss3;
//		ss3 << int(fps);
//
//		//show the current frame and the fg masks
//		stringstream ss1; ss1 << score;
//		cv::putText(ImRGBRes, ss1.str(), cv::Point(mag * (cx - rw / 2), mag * (cy - rh / 2 - 5)), FONT_HERSHEY_SIMPLEX, 0.5, cvScalar(255, 0, 255), 1, 8, false);
//		cv::putText(ImRGBRes, "Tracker: " + ss3.str() + "fps", Point(5, 40), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(29, 45, 255));
//		cv::imshow("Tracker", ImRGBRes);
//		stringstream ostr; ostr << frameNumberString.c_str(); string nums; ostr >> nums;
//		//string imageToSave = "Frame/frame_" + nums + ".png";
//		//cv::imwrite(imageToSave, ImRGBRes);
//		keyboard = waitKey(1);
//
//
//		// save to file current bb
//		rects << l << ' ' << t << ' ' << rw << ' ' << rh << '\n';
//		
//
//		if (count == counter) break;
//
//		ostringstream oss;
//		oss << setfill('0') << setw(nzeros) << (frameNumber + 1);
//		string nextFrameNumberString = oss.str();
//		string nextFrameFilename = prefix + nextFrameNumberString + suffix;
//		if (use_gray == true)
//			Im = imread(nextFrameFilename, CV_LOAD_IMAGE_GRAYSCALE);
//		else
//			Im = imread(nextFrameFilename);
//		if (Im.empty()) { cerr << "Unable to open image frame: " << nextFrameFilename << endl; break; }
//		else
//			fn.assign(nextFrameFilename);
//	}
//
//	avgfps /= (counter-1);
//	fpsfile << minfps << ' ' << avgfps << ' ' << maxfps << '\n';
//
//	rects.close();
//	fpsfile.close();
//
//	ImRGBRes.release(); Im.release();
//	delete[] dataB; delete[] dataYorR; delete[] dataG;
//}
//
//
