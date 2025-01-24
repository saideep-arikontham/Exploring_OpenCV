/*
Saideep Arikontham
January 2025
CS 5330 OpenCV tutorial
*/

#include <cmath>
#include <cstdio>
#include "filters.h"
#include <algorithm> // gives std::max
#include "faceDetect.h"
#include <onnxruntime_cxx_api.h>
#include "DA2Network.hpp"


// Filter to apply simple gray filter
// Press 'g' to change video to gray scale
cv::Mat gray1_filter(cv::Mat &frame){
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    return gray;
}


// Filter to apply another version of gray filter
// Press 'h' to change video to another gray scale
cv::Mat gray2_filter(cv::Mat &frame){
    cv::Mat dst;

    frame.copyTo( dst );
    for (int i=0; i < dst.rows; i++){
        cv::Vec3b *ptr = dst.ptr<cv::Vec3b>(i); // get the pointer for the row i data

        for( int j=0; j<dst.cols; j++){
            float temp = std::max(ptr[j][0], std::max(ptr[j][1], ptr[j][2])) - std::min(ptr[j][0], std::min(ptr[j][1], ptr[j][2]))/2;
            ptr[j][2] = temp ;
            ptr[j][0] = temp ;
            ptr[j][1] = temp ; 
        }
    }
    return dst;
}


// Filter to apply sepia filter
// Press 'a' to change video to sepia tone
cv::Mat sepia_filter(cv::Mat &frame){

    cv::Mat dst;
    frame.copyTo( dst );

    for (int i=0; i<dst.rows; i++){
        cv::Vec3b *ptr = dst.ptr<cv::Vec3b>(i);
        
        for (int j=0; j<dst.cols; j++){
            float temp_blue = 0.272*ptr[j][2] + 0.534*ptr[j][1] + 0.131*ptr[j][0];    // Blue coefficients for R, G, B  (pay attention to the channel order)
            float temp_green = (0.349*ptr[j][2] + 0.686*ptr[j][1] + 0.168*ptr[j][0]);    // Green coefficients
            float temp_red = (0.393*ptr[j][2] + 0.769*ptr[j][1] + 0.189*ptr[j][0]);    // Red coefficients
            
            ptr[j][2] = std::min(255.0f, temp_red);
            ptr[j][1] = std::min(255.0f, temp_green);
            ptr[j][0] = std::min(255.0f, temp_blue);
        }
    }
    return dst;
}


// Filter to apply gaussian blur with 5x5 kernel
// Press 'b' to blur the video - Would not work as separable filters implementation is used.
int blur5x5_1(cv::Mat &src, cv::Mat &dst){
    // Gaussian 5x5 kernel (integer approximation)
    int kernel[5][5] = {
        {1, 2, 4, 2, 1},
        {2, 4, 8, 4, 2},
        {4, 8, 16, 8, 4},
        {2, 4, 8, 4, 2},
        {1, 2, 4, 2, 1}
    };

    src.copyTo( dst );

    for (int i=2; i<dst.rows-2; i++){
        cv::Vec3b *ptr = dst.ptr<cv::Vec3b>(i);

        for (int j=2; j<dst.cols-2; j++){
            int blue = 0, green = 0, red = 0;

            blue = ptr[j-2][0]*kernel[0][0] + ptr[j-1][0]*kernel[0][1] + ptr[j][0]*kernel[0][2] + ptr[j+1][0]*kernel[0][3] + ptr[j+2][0]*kernel[0][4] +
                   ptr[j-2][0]*kernel[1][0] + ptr[j-1][0]*kernel[1][1] + ptr[j][0]*kernel[1][2] + ptr[j+1][0]*kernel[1][3] + ptr[j+2][0]*kernel[1][4] +
                   ptr[j-2][0]*kernel[2][0] + ptr[j-1][0]*kernel[2][1] + ptr[j][0]*kernel[2][2] + ptr[j+1][0]*kernel[2][3] + ptr[j+2][0]*kernel[2][4] +
                   ptr[j-2][0]*kernel[3][0] + ptr[j-1][0]*kernel[3][1] + ptr[j][0]*kernel[3][2] + ptr[j+1][0]*kernel[3][3] + ptr[j+2][0]*kernel[3][4] +
                   ptr[j-2][0]*kernel[4][0] + ptr[j-1][0]*kernel[4][1] + ptr[j][0]*kernel[4][2] + ptr[j+1][0]*kernel[4][3] + ptr[j+2][0]*kernel[4][4];

            green = ptr[j-2][1]*kernel[0][0] + ptr[j-1][1]*kernel[0][1] + ptr[j][1]*kernel[0][2] + ptr[j+1][1]*kernel[0][3] + ptr[j+2][1]*kernel[0][4] +
                    ptr[j-2][1]*kernel[1][0] + ptr[j-1][1]*kernel[1][1] + ptr[j][1]*kernel[1][2] + ptr[j+1][1]*kernel[1][3] + ptr[j+2][1]*kernel[1][4] +
                    ptr[j-2][1]*kernel[2][0] + ptr[j-1][1]*kernel[2][1] + ptr[j][1]*kernel[2][2] + ptr[j+1][1]*kernel[2][3] + ptr[j+2][1]*kernel[2][4] +
                    ptr[j-2][1]*kernel[3][0] + ptr[j-1][1]*kernel[3][1] + ptr[j][1]*kernel[3][2] + ptr[j+1][1]*kernel[3][3] + ptr[j+2][1]*kernel[3][4] +
                    ptr[j-2][1]*kernel[4][0] + ptr[j-1][1]*kernel[4][1] + ptr[j][1]*kernel[4][2] + ptr[j+1][1]*kernel[4][3] + ptr[j+2][1]*kernel[4][4];

            red = ptr[j-2][2]*kernel[0][0] + ptr[j-1][2]*kernel[0][1] + ptr[j][2]*kernel[0][2] + ptr[j+1][2]*kernel[0][3] + ptr[j+2][2]*kernel[0][4] +
                  ptr[j-2][2]*kernel[1][0] + ptr[j-1][2]*kernel[1][1] + ptr[j][2]*kernel[1][2] + ptr[j+1][2]*kernel[1][3] + ptr[j+2][2]*kernel[1][4] +
                  ptr[j-2][2]*kernel[2][0] + ptr[j-1][2]*kernel[2][1] + ptr[j][2]*kernel[2][2] + ptr[j+1][2]*kernel[2][3] + ptr[j+2][2]*kernel[2][4] +
                  ptr[j-2][2]*kernel[3][0] + ptr[j-1][2]*kernel[3][1] + ptr[j][2]*kernel[3][2] + ptr[j+1][2]*kernel[3][3] + ptr[j+2][2]*kernel[3][4] +
                  ptr[j-2][2]*kernel[4][0] + ptr[j-1][2]*kernel[4][1] + ptr[j][2]*kernel[4][2] + ptr[j+1][2]*kernel[4][3] + ptr[j+2][2]*kernel[4][4];

            ptr[j][0] = blue/100;
            ptr[j][1] = green/100;
            ptr[j][2] = red/100;
        }
    }
    return 0;
}


// Filter to apply gaussian blur with separable kernels
// Press 'b' to blur the video
int blur5x5_2( cv::Mat &src, cv::Mat &dst ){
    // separable filter
    int kernel[5] = {1, 2, 4, 2, 1};

    // https://stackoverflow.com/questions/7765142/gaussian-blur-and-fft -  Why do we need temporary buffer?
    cv::Mat temp;
    src.copyTo(temp);
    src.copyTo( dst );

    // Horizontal
    for (int i=0; i<src.rows; i++){
        cv::Vec3b *srcPtr = src.ptr<cv::Vec3b>(i);
        cv::Vec3b *tempPtr = temp.ptr<cv::Vec3b>(i);

        for (int j=2; j<src.cols-2; j++){
            int blue = 0, green = 0, red = 0;

            blue = srcPtr[j-2][0]*kernel[0] + srcPtr[j-1][0]*kernel[1] + srcPtr[j][0]*kernel[2] + srcPtr[j+1][0]*kernel[3] + srcPtr[j+2][0]*kernel[4];
            green = srcPtr[j-2][1]*kernel[0] + srcPtr[j-1][1]*kernel[1] + srcPtr[j][1]*kernel[2] + srcPtr[j+1][1]*kernel[3] + srcPtr[j+2][1]*kernel[4];
            red = srcPtr[j-2][2]*kernel[0] + srcPtr[j-1][2]*kernel[1] + srcPtr[j][2]*kernel[2] + srcPtr[j+1][2]*kernel[3] + srcPtr[j+2][2]*kernel[4];

            tempPtr[j][0] = blue/10;
            tempPtr[j][1] = green/10;
            tempPtr[j][2] = red/10;
        }
    }

    // Vertical
    for (int i=2; i<src.rows-2; i++){
        for (int j=0; j<src.cols; j++){
            int blue = 0, green = 0, red = 0;

            blue = temp.at<cv::Vec3b>(i - 2, j)[0] * kernel[0] + temp.at<cv::Vec3b>(i - 1, j)[0] * kernel[1] + temp.at<cv::Vec3b>(i, j)[0] * kernel[2] + temp.at<cv::Vec3b>(i + 1, j)[0] * kernel[3] + temp.at<cv::Vec3b>(i + 2, j)[0] * kernel[4];
            green = temp.at<cv::Vec3b>(i - 2, j)[1] * kernel[0] + temp.at<cv::Vec3b>(i - 1, j)[1] * kernel[1] + temp.at<cv::Vec3b>(i, j)[1] * kernel[2] + temp.at<cv::Vec3b>(i + 1, j)[1] * kernel[3] + temp.at<cv::Vec3b>(i + 2, j)[1] * kernel[4];
            red = temp.at<cv::Vec3b>(i - 2, j)[2] * kernel[0] + temp.at<cv::Vec3b>(i - 1, j)[2] * kernel[1] + temp.at<cv::Vec3b>(i, j)[2] * kernel[2] + temp.at<cv::Vec3b>(i + 1, j)[2] * kernel[3] + temp.at<cv::Vec3b>(i + 2, j)[2] * kernel[4];

            dst.at<cv::Vec3b>(i, j)[0] = blue/10;
            dst.at<cv::Vec3b>(i, j)[1] = green/10;
            dst.at<cv::Vec3b>(i, j)[2] = red/10;
        }
    }

    return 0;
}


// Sobel filter for X direction
// Press 'x' to get the sobel filter applied frame
int sobelX3x3( cv::Mat &src, cv::Mat &dst ){
    // separable filter
    int horizontal[3] = {-1, 0, 1};
    int vertical[3] = {1, 2, 1};
    cv::Mat temp;

    // copying src size to dst and temp with 16 bit signed 3 channel
    dst = cv::Mat::zeros(src.size(), CV_16SC3);
    temp = cv::Mat::zeros(src.size(), CV_16SC3);

    // Horizontal
    for (int i=0; i<src.rows; i++){
        cv::Vec3b *src_ptr = src.ptr<cv::Vec3b>(i);
        cv::Vec3s *temp_ptr = temp.ptr<cv::Vec3s>(i);
        for (int j=1; j<src.cols-1; j++){
            int blue = 0, green = 0, red = 0;

            blue = src_ptr[j-1][0]*horizontal[0] + src_ptr[j][0]*horizontal[1] + src_ptr[j+1][0]*horizontal[2];
            green = src_ptr[j-1][1]*horizontal[0] + src_ptr[j][1]*horizontal[1] + src_ptr[j+1][1]*horizontal[2];
            red = src_ptr[j-1][2]*horizontal[0] + src_ptr[j][2]*horizontal[1] + src_ptr[j+1][2]*horizontal[2];

            temp_ptr[j][0] = blue;
            temp_ptr[j][1] = green;
            temp_ptr[j][2] = red;
        }
    }

    // Vertical
    for (int i=1; i<src.rows-1; i++){
        for (int j=0; j<src.cols; j++){
            int blue = 0, green = 0, red = 0;

            blue = temp.at<cv::Vec3s>(i - 1, j)[0] * vertical[0] + temp.at<cv::Vec3s>(i, j)[0] * vertical[1] + temp.at<cv::Vec3s>(i + 1, j)[0] * vertical[2];
            green = temp.at<cv::Vec3s>(i - 1, j)[1] * vertical[0] + temp.at<cv::Vec3s>(i, j)[1] * vertical[1] + temp.at<cv::Vec3s>(i + 1, j)[1] * vertical[2];
            red = temp.at<cv::Vec3s>(i - 1, j)[2] * vertical[0] + temp.at<cv::Vec3s>(i, j)[2] * vertical[1] + temp.at<cv::Vec3s>(i + 1, j)[2] * vertical[2];

            dst.at<cv::Vec3s>(i, j)[0] = blue;
            dst.at<cv::Vec3s>(i, j)[1] = green;
            dst.at<cv::Vec3s>(i, j)[2] = red;
        }
    }

    return 0;
}


// Sobel filter for Y direction
// Press 'y' to get the sobel filter applied frame
int sobelY3x3( cv::Mat &src, cv::Mat &dst ) {
    // Separable filter
    int horizontal[3] = {1, 2, 1};
    int vertical[3] = {1, 0, -1};
    cv::Mat temp;

    // Initialize temp and dst with CV_16SC3 (16-bit signed, 3 channels)
    dst = cv::Mat::zeros(src.size(), CV_16SC3);
    temp = cv::Mat::zeros(src.size(), CV_16SC3);

    // Horizontal pass
    for (int i = 0; i < src.rows; i++) {
        cv::Vec3b *src_ptr = src.ptr<cv::Vec3b>(i);
        cv::Vec3s *temp_ptr = temp.ptr<cv::Vec3s>(i);

        for (int j = 1; j < src.cols - 1; j++) {
            int blue = 0, green = 0, red = 0;

            blue = src_ptr[j - 1][0] * horizontal[0] + src_ptr[j][0] * horizontal[1] + src_ptr[j + 1][0] * horizontal[2];
            green = src_ptr[j - 1][1] * horizontal[0] + src_ptr[j][1] * horizontal[1] + src_ptr[j + 1][1] * horizontal[2];
            red = src_ptr[j - 1][2] * horizontal[0] + src_ptr[j][2] * horizontal[1] + src_ptr[j + 1][2] * horizontal[2];

            temp_ptr[j][0] = blue;
            temp_ptr[j][1] = green;
            temp_ptr[j][2] = red;
        }
    }

    // Vertical pass
    for (int i = 1; i < src.rows - 1; i++) {
        for (int j = 0; j < src.cols; j++) {
            int blue = 0, green = 0, red = 0;

            blue = temp.at<cv::Vec3s>(i - 1, j)[0] * vertical[0] + temp.at<cv::Vec3s>(i, j)[0] * vertical[1] + temp.at<cv::Vec3s>(i + 1, j)[0] * vertical[2];
            green = temp.at<cv::Vec3s>(i - 1, j)[1] * vertical[0] + temp.at<cv::Vec3s>(i, j)[1] * vertical[1] + temp.at<cv::Vec3s>(i + 1, j)[1] * vertical[2];
            red = temp.at<cv::Vec3s>(i - 1, j)[2] * vertical[0] + temp.at<cv::Vec3s>(i, j)[2] * vertical[1] + temp.at<cv::Vec3s>(i + 1, j)[2] * vertical[2];


            dst.at<cv::Vec3s>(i, j)[0] = blue;
            dst.at<cv::Vec3s>(i, j)[1] = green;
            dst.at<cv::Vec3s>(i, j)[2] = red;
        }
    }

    return 0;
}


// Magnitude of the sobel output
// Press 'm' to get the magnitude computed frame
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst) {

    dst = cv::Mat::zeros(sx.size(), CV_8UC3);

    // Compute gradient magnitude for each channel
    for (int i = 0; i < sx.rows; i++) {
        cv::Vec3s *sx_ptr = sx.ptr<cv::Vec3s>(i);
        cv::Vec3s *sy_ptr = sy.ptr<cv::Vec3s>(i);
        cv::Vec3b *dst_ptr = dst.ptr<cv::Vec3b>(i);

        for (int j = 0; j < sx.cols; j++) {
            float blue_mag = std::sqrt(sx_ptr[j][0] * sx_ptr[j][0] + sy_ptr[j][0] * sy_ptr[j][0]);
            float green_mag = std::sqrt(sx_ptr[j][1] * sx_ptr[j][1] + sy_ptr[j][1] * sy_ptr[j][1]);
            float red_mag = std::sqrt(sx_ptr[j][2] * sx_ptr[j][2] + sy_ptr[j][2] * sy_ptr[j][2]);

            dst_ptr[j][0] = std::min(255.0f, blue_mag);
            dst_ptr[j][1] = std::min(255.0f, green_mag);
            dst_ptr[j][2] = std::min(255.0f, red_mag);
        }
    }

    return 0; 
}


// Blur and quantize the image
// Press 'l' to get the quantized frame
int blurQuantize( cv::Mat &src, cv::Mat &dst, int levels ){
    
    // Blur the image
    blur5x5_2(src, dst);

    // Calculating b
    int b = 255/levels;

    // Quantize the image
    for (int i=0; i<dst.rows; i++){
        cv::Vec3b *ptr = dst.ptr<cv::Vec3b>(i);
        int xt;
        for (int j=0; j<dst.cols; j++){
            xt = ptr[j][0] / b;
            ptr[j][0] = xt * b;

            xt = ptr[j][1] / b;
            ptr[j][1] = xt * b;

            xt = ptr[j][2] / b;
            ptr[j][2] = xt * b;
        }
    }
    return 0;
}


// Filter to get the face box
// Press 'f' to get the face box for the video
cv::Mat get_face_box(cv::Mat &frame){
    //detect faces
    cv::Mat grey;
    std::vector<cv::Rect> faces;
    cv::Rect last(0, 0, 0, 0);

    // convert the image to greyscale
    cv::cvtColor( frame, grey, cv::COLOR_BGR2GRAY, 0);

    // detect faces
    detectFaces( grey, faces );

    // draw boxes around the faces
    drawBoxes( frame, faces);

    // add a little smoothing by averaging the last two detections
    if( faces.size() > 0 ) {
    last.x = (faces[0].x + last.x)/2;
    last.y = (faces[0].y + last.y)/2;
    last.width = (faces[0].width + last.width)/2;
    last.height = (faces[0].height + last.height)/2;
    }
    return frame;
}


// Function to get the depth information
cv::Mat get_depth(float scale_factor, cv::Mat &src){
    cv::Mat dst;

    // load model
    DA2Network da_net( "./src/model_fp16.onnx" );

    // set the network input
    da_net.set_input( src, scale_factor );

    // run the network
    da_net.run_network( dst, src.size() );

    return dst;
}


// Function to apply background blur
// Press 'd' to get the frame with background blur effect
cv::Mat background_blur(cv::Mat &frame, cv::Mat &dst_vis, float scale_factor){
    cv::Mat dst = get_depth(scale_factor, frame);

    // depth image
    cv::normalize(dst, dst_vis, 0, 255, cv::NORM_MINMAX, CV_8UC1);

    // identify if pixel is part of background or not based on threshold
    cv::Mat background_mask;
    cv::threshold(dst_vis, background_mask, 70, 255, cv::THRESH_BINARY); // pixel value above 70 set to 255 (white) and below 70 set to 0 (black)
    // cv::imshow("bgmask", background_mask);

    // apply intensive blur (not using 5x5 blur code that I wrote because its less intense)
    cv::Mat blurred_frame;
    cv::GaussianBlur(frame, blurred_frame, cv::Size(27, 27), 0);

    cv::Mat result;
    frame.copyTo(result);
    // combine background and foreground
    for (int i=0; i<frame.rows; i++){
        for (int j=0; j<frame.cols; j++){
            if(background_mask.at<uchar>(i, j) == 0){
                result.at<cv::Vec3b>(i, j) = blurred_frame.at<cv::Vec3b>(i, j);
            }
        }
    }
    return result;
}



// ADDITIONAL TASKS

// Filter to apply mirror effect
// Press 'u' to get the mirrored video
int mirror_filter(cv::Mat &src, cv::Mat &dst){

    cv::Mat temp;
    src.copyTo( dst );
    src.copyTo( temp );

    for (int i=0; i<dst.rows; i++){
        cv::Vec3b *ptr = dst.ptr<cv::Vec3b>(i);
        for (int j=0; j<dst.cols; j++){
            ptr[j][0] = temp.at<cv::Vec3b>(i, dst.cols-j-1)[0];
            ptr[j][1] = temp.at<cv::Vec3b>(i, dst.cols-j-1)[1];
            ptr[j][2] = temp.at<cv::Vec3b>(i, dst.cols-j-1)[2];
        }
    }
    return 0;
}


// Filter to apply median filter
// Press 'i' to get the video with median filter
int medianFilter5x5(cv::Mat &frame, cv::Mat &dst){
    cv::Mat padded_frame;
    cv::copyMakeBorder(frame, padded_frame, 2, 2, 2, 2, cv::BORDER_REFLECT);

    frame.copyTo(dst);

    for (int i=2; i<frame.rows-2; i++){
        for(int j=2; j<frame.cols-2; j++){
            for (int c=0; c<3; c++){
                int pixels[25] = {
                    padded_frame.at<cv::Vec3b>(i-2, j-2)[c], padded_frame.at<cv::Vec3b>(i-2, j-1)[c], padded_frame.at<cv::Vec3b>(i-2, j)[c], padded_frame.at<cv::Vec3b>(i-2, j+1)[c], padded_frame.at<cv::Vec3b>(i-2, j+2)[c],
                    padded_frame.at<cv::Vec3b>(i-1, j-2)[c], padded_frame.at<cv::Vec3b>(i-1, j-1)[c], padded_frame.at<cv::Vec3b>(i-1, j)[c], padded_frame.at<cv::Vec3b>(i-1, j+1)[c], padded_frame.at<cv::Vec3b>(i-1, j+2)[c],
                    padded_frame.at<cv::Vec3b>(i, j-2)[c], padded_frame.at<cv::Vec3b>(i, j-1)[c], padded_frame.at<cv::Vec3b>(i, j)[c], padded_frame.at<cv::Vec3b>(i, j+1)[c], padded_frame.at<cv::Vec3b>(i, j+2)[c],
                    padded_frame.at<cv::Vec3b>(i+1, j-2)[c], padded_frame.at<cv::Vec3b>(i+1, j-1)[c], padded_frame.at<cv::Vec3b>(i+1, j)[c], padded_frame.at<cv::Vec3b>(i+1, j+1)[c], padded_frame.at<cv::Vec3b>(i+1, j+2)[c],
                    padded_frame.at<cv::Vec3b>(i+2, j-2)[c], padded_frame.at<cv::Vec3b>(i+2, j-1)[c], padded_frame.at<cv::Vec3b>(i+2, j)[c], padded_frame.at<cv::Vec3b>(i+2, j+1)[c], padded_frame.at<cv::Vec3b>(i+2, j+2)[c]
                };
                std::sort(pixels, pixels+25);
                dst.at<cv::Vec3b>(i-2, j-2)[c] = pixels[12];
            }
        }
    }
    return 0;
}



// Filter to apply fog effect
// Press 'o' to get the frame with fog effect
cv::Mat background_fog(cv::Mat &frame, cv::Mat &dst, float scale_factor){

    dst = get_depth(scale_factor, frame);
    int fog_color[3] = {179, 179, 242}; 

    // Ensure depth is normalized to [0, 1]
    cv::Mat normalized_depth;
    cv::normalize(dst, normalized_depth, 0, 1, cv::NORM_MINMAX, CV_32FC1);

    cv::Mat result;
    frame.copyTo(result);

    // Apply fog effect
    for (int i = 0; i < frame.rows; i++) {
        for (int j = 0; j < frame.cols; j++) {

            // Get inverted depth value, to make farther objects appear foggy
            float depth_value = normalized_depth.at<float>(i, j);
            depth_value = std::exp(-depth_value * 3.0f); // Exponential decay

            // Blend the fog color with the original pixel color
            result.at<cv::Vec3b>(i, j)[0] = (1 - depth_value) * result.at<cv::Vec3b>(i, j)[0] + depth_value * fog_color[0];
            result.at<cv::Vec3b>(i, j)[1] = (1 - depth_value) * result.at<cv::Vec3b>(i, j)[1] + depth_value * fog_color[1];
            result.at<cv::Vec3b>(i, j)[2] = (1 - depth_value) * result.at<cv::Vec3b>(i, j)[2] + depth_value * fog_color[2];
        }
    }
    return result;
}


// EXTENTIONS

// Filter to apply a sketch type effect (Atleast that was the goal)
// Press 'p' to get the frame with sketch effect
cv::Mat sketch_filter(cv::Mat &frame){

    cv::Mat temp1;
    blurQuantize(frame, temp1, 25);

    cv::Mat sx;
    sobelX3x3(frame, sx);

    cv::Mat sy;
    sobelY3x3(frame, sy);

    cv::Mat temp2;
    magnitude(sx, sy, temp2);

    cv::cvtColor(temp2, temp2, cv::COLOR_BGR2GRAY);
    for(int i=0; i<temp2.rows; i++){
        for(int j=0; j<temp2.cols; j++){
            if(temp2.at<uchar>(i, j) <= 50){
                temp2.at<uchar>(i, j) = 255;
            }
            else{
                temp2.at<uchar>(i, j) = 0;
            }
        }
    }

    cv::Mat temp3;
    medianBlur(temp2, temp3, 5);
    //medianFilter5x5(temp2, temp3);

    cv::Mat dst;
    temp1.copyTo( dst ); 

    for(int i=0; i<temp1.rows; i++){
        for(int j=0; j<temp1.cols; j++){
            if(temp3.at<uchar>(i, j) == 0){
                dst.at<cv::Vec3b>(i, j)[0] = 0;
                dst.at<cv::Vec3b>(i, j)[1] = 0;
                dst.at<cv::Vec3b>(i, j)[2] = 0;
            }
        }
    } 
    return dst;
}


// Filter to create a bouncing circle 
// Press 'k' to get the video stream with bouncing circle
int bouncing_circle(cv::Mat &frame) {
    // defining static variables to keep track of the circle's position in between function calls.

    // initial position
    static int x = frame.cols / 2; 
    static int y = frame.rows / 2; 
    // position change
    static int dx = 10; 
    static int dy = 10;

    int radius = 100;

    // Update the circle's position
    x += dx;
    y += dy;

    // Conditions to keep circle within the frame
    if (x - radius < 0 || x + radius >= frame.cols) { // taking radius into account will help to keep the circle completely within the window
        dx = -dx + (std::rand() % 3 - 1) * 2; // change to opposite direction with randomness
    }
    if (y - radius < 0 || y + radius >= frame.rows) {
        dy = -dy + (std::rand() % 3 - 1) * 2;
    }

    // if dx or dy becomes 0 from the above conditions, then reset the values to 0
    if (dx == 0)
        dx = 10;
    if (dy == 0) 
        dy = 10;

    // Circle
    cv::circle(frame, cv::Point(x, y), radius, cv::Scalar(230, 123, 41), -1);

    return 0;
}