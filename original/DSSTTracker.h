
// This is a C++ implementation of the Discriminative Scale Space Tracker (DSST) [1]
// The implementation is built upon the pure Matlab code provided by [1]. 
// The code provided by [2] is used for computing the HOG features and rescaling image.
//
// [1] Martin Danelljan, Gustav Huger, Fahad Shahbaz Khan and Michael Felsberg.
//     "Accurate Scale Estimation for Robust Visual Tracking".
//     Proceedings of the British Machine Vision Conference(BMVC), 2014,
//     Winner of VOT 2014 challenge.
//
// [2] Piotr Dollar.
//	   "PiotrТs Image and Video Matlab Toolbox (PMT)."
//	   http://vision.ucsd.edu/~pdollar/toolbox/doc/index.html.
//*******************************************************************************
//
//                         Copyright 2015 Sersid :D
// 
//*******************************************************************************
#pragma once
#include <fftw3.h>

namespace ctr { // correlation tracker namespace

	class dsst_tracker
	{

	public:

		dsst_tracker(float _padding = 1.0f, float _output_sigma_factor = 0.0625f,
			float _scale_sigma_factor = 0.25f, float _lambda = 0.01f,
			float _learning_rate = 0.0250f, int _number_of_scales = 33,
			float _scale_step = 1.02f, int _scale_model_max_area = 512, bool _use_scale_4_translation_estimate = true);

		~dsst_tracker();

		bool initializeTargetModel(int c_x, int c_y, int t_w, int t_h, int _imw, int _imh, unsigned char* dataYorR, unsigned char* dataG = 0, unsigned char* dataB = 0);
		bool findNextLocation(unsigned char* dataYorR, unsigned char* dataG = 0, unsigned char* dataB = 0);
		bool getNewLocationCoordinates(int &x, int &y, int &w, int &h, float &scr);

	private:
	
		// параметры настройки
		float padding;             // extra area surrounding the target
		float output_sigma_factor; // standard deviation for the desired translation filter output
		float scale_sigma_factor;  // standard deviation for the desired scale filter output
		float lambda;              // regularization weight (denoted "lambda" in the paper)
		float learning_rate;       // tracking model learning rate(denoted "eta" in the paper)
		int nScales;               // number of scale levels(denoted "S" in the paper)
		float scale_step;          // Scale increment factor(denoted "a" in the paper)
		int scale_model_max_area;  // the maximum size of scale examples
		bool use_scale_4_translation_estimate; // использовать ли оценку масштаба при оценке сдвига
		int imw;                   // размер изображени€ по ширине (рекомендуемое дл€ производительности 320)
		int imh;                   // размер изображени€ по высоте (рекомендуемое дл€ производительности 240)
		int ix;                    // центр объекта по оси х
		int iy;                    // центр объекта по оси y
		int iw;                    // ширина рамки вокруг объекта
		int ih;                    // высота рамки вокруг объекта
		int base_target_sz[2];     // h,w - target size att scale = 1
		int target_size[2];        // h,w - target size att scale = 1
		int sz[2];                 // window size, taking padding into account
		int scale_model_sz[2];     // h,w
		float *scaleFactors;       // возможные увеличени€ относительно исходного размера объекта
		float currentScaleFactor;  // текущее увеличение относительно исходного размера объекта
		float min_scale_factor;    // минимально допустимое увеличение относительно исходного размера объекта
		float max_scale_factor;    // максимально допустимое увеличение относительно исходного размера объекта
		float  *scale_window;      // 1D
		float  *cos_window;        // 2D - rows-major order
		float  score;              // оценка трекером качествао прослеживани€ [0..1], example, 1 - good, 0 - bad
		
		//вспомогательные функции
		void make_translation_hann_window(int n, float* window);
		void make_scale_hann_window(int n, float *window);
		void make_scale_cosine_mask();
		void make_translation_cosine_mask();
		void get_translation_feature_map(float *In, float **Out, int h, int w, int din);
		void get_scale_feature_map(float *In, float *temp, int h, int w, int din, float **scale_sample, int s);
		void get_translation_sample(unsigned char* dataYorR, unsigned char* dataG, unsigned char* dataB, float **sample); // data in row-major order
		void get_scale_sample(unsigned char* dataYorR, unsigned char* dataG, unsigned char* dataB, float **scale_sample);
		void extract_training_sample_info(unsigned char* dataYorR, unsigned char* dataG, unsigned char* dataB);
		void extract_translation_test_sample(unsigned char* dataYorR, unsigned char* dataG, unsigned char* dataB);
		void extract_scale_test_sample(unsigned char* dataYorR, unsigned char* dataG, unsigned char* dataB);
		void dsst_tracker::extract_image(unsigned char* dataYorR, unsigned char* dataG, unsigned char* dataB,
			                             int pw, int ph, int xm, int xp, int ym, int yp, float *im_patch, int d);

		// вспомогательные переменные
		fftwf_plan pyf;
		fftwf_plan pysf;
		fftwf_complex *yf;
		fftwf_complex *ysf;
		int dout;
		fftwf_plan *pxtf;
		fftwf_complex *rt;           // ratio for translation
		float *respt;
		fftwf_plan prespt;
		fftwf_complex **new_hf_num;
		float *new_hf_den;
		fftwf_complex **hf_num;
		float *hf_den;
		float **xt;                 // translation test sample
		fftwf_complex **xtf;        // fourier translation test sample 
		int dscale;
		int sizess;
		float *temps;
		float **xs;                 // translation train sample (data in row-major order созданные по правилам fftw)
		fftwf_complex **new_sf_num;
		fftwf_complex **sf_num;
		float *new_sf_den;
		float *sf_den;
		fftwf_complex *rts;         // ratio for scale
		fftwf_plan presps;
		float *resps;
		fftwf_complex** xsf;
		fftwf_plan *pxsf;
		int prodsz;
		int prodszhalf;
	};

}

