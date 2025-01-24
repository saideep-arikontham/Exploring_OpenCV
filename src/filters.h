/*
Saideep Arikontham
January 2025
CS 5330 OpenCV tutorial
*/

#ifndef FILTERS_H
#define FILTERS_H

#include <opencv2/opencv.hpp>

// Function declaration for gray1_filter
cv::Mat gray1_filter(cv::Mat &frame);
cv::Mat gray2_filter(cv::Mat &frame);
cv::Mat sepia_filter(cv::Mat &frame);
int blur5x5_1(cv::Mat &src, cv::Mat &dst);
int blur5x5_2( cv::Mat &src, cv::Mat &dst );
int sobelX3x3( cv::Mat &src, cv::Mat &dst );
int sobelY3x3( cv::Mat &src, cv::Mat &dst );
int magnitude( cv::Mat &sx, cv::Mat &sy, cv::Mat &dst );
int blurQuantize( cv::Mat &src, cv::Mat &dst, int levels );
cv::Mat get_face_box(cv::Mat &frame);
cv::Mat get_depth(float scale_factor, cv::Mat &src);
cv::Mat background_blur(cv::Mat &frame, cv::Mat &dst_vis, float scale_factor);

int mirror_filter(cv::Mat &src, cv::Mat &dst);
int medianFilter5x5(cv::Mat &frame, cv::Mat &dst);
cv::Mat background_fog(cv::Mat &frame, cv::Mat &dst, float scale_factor);
cv::Mat sketch_filter(cv::Mat &frame);
int bouncing_circle(cv::Mat &frame);


#endif // FILTERS_H