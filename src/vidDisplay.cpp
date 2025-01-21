/*
Saideep Arikontham
January 2025
CS 5330 OpenCV tutorial
*/

#include <cstdio> // gives printf
#include <cstring> // gives strcpy
#include <opencv2/opencv.hpp> // openCV
#include <algorithm> // gives std::max
#include "filters.h"


int main(int argc, char *argv[]) {
        cv::VideoCapture *capdev;
    
        // open the video device
        capdev = new cv::VideoCapture(0);
        if( !capdev->isOpened() ) {
                printf("Unable to open video device\n");
                return(-1);
        }

        // get some properties of the image
        cv::Size refS( (int) capdev->get(cv::CAP_PROP_FRAME_WIDTH ),
                       (int) capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
        printf("Expected size: %d %d\n", refS.width, refS.height);

        cv::namedWindow("Video", 1); // identifies a window
        cv::Mat frame;

       char prev_key = ' ';

        for(;;) {
                *capdev >> frame; // get a new frame from the camera, treat as a stream
                if( frame.empty() ) {
                  printf("frame is empty\n");
                  break;
                }                

                // Checking key press for Continuous effects - video filters
                if( prev_key == 'g' ){
                    cv::Mat gray = gray1_filter(frame);
                    cv::imshow("Video", gray);
                }
                else if( prev_key == 'h' ){
                    cv::Mat gray = gray2_filter(frame);
                    cv::imshow("Video", gray);
                }
                else if( prev_key == 'a'){
                    cv::Mat sepia = sepia_filter(frame);
                    cv::imshow("Video", sepia);
                }
                else if( prev_key == 'b'){
                    cv::Mat dst;
                    blur5x5_2(frame, dst);
                    cv::imshow("Video", dst);
                }
                else if( prev_key == 'f'){
                    frame = get_face_box(frame);
                    cv::imshow("Video", frame);
                }
                else if(prev_key == 'p'){
                    cv::Mat dst;
                    mirror_filter(frame, dst);
                    cv::imshow("Video", dst);
                }
                else{
                    cv::imshow("Video", frame);
                }

                // see if there is a waiting keystroke
                char key = cv::waitKey(10);

                // Checking key press for one time effects
                if( key == 'q') {
                    break;
                }
                else if(key == 'x'){
                    cv::Mat dst;
                    sobelX3x3(frame, dst);
                    cv::convertScaleAbs(dst, dst); 
                    cv::imshow("Sobel X", dst);
                }
                else if(key == 'y'){
                    cv::Mat dst;
                    sobelY3x3(frame, dst);
                    cv::convertScaleAbs(dst, dst); 
                    cv::imshow("Sobel Y", dst);
                }
                else if(key == 'm'){
                    cv::Mat sx, sy, dst;
                    cv::imwrite("output/sobel_magnitude_original.jpg", frame);
                    sobelX3x3(frame, sx);
                    cv::imwrite("output/sobel_x.jpg", sx);
                    sobelY3x3(frame, sy);
                    cv::imwrite("output/sobel_y.jpg", sy);
                    magnitude(sx, sy, dst);
                    
                    // cv::convertScaleAbs(dst, dst); 
                    cv::imshow("Sobel Magnitude", dst);
                    cv::imwrite("output/sobel_magnitude.jpg", dst);
                }
                else if(key == 'l'){
                    cv::Mat dst;
                    cv::imwrite("output/blurQuantize_original.jpg", frame);
                    blurQuantize(frame, dst, 10);
                    cv::imshow("Quantized", dst);
                    cv::imwrite("output/blurQuantize.jpg", dst);
                }
                else if(key == 'd'){
                    const float reduction = 0.5;
                    float scale_factor = 256.0 / (refS.height*reduction);
                    printf("Using scale factor %.2f\n", scale_factor);

                    cv::imwrite("output/portrait_original.jpg", frame);
                    cv::Mat result = background_blur(frame, scale_factor);
                    cv::imwrite("output/portrait_image.jpg", result);
                }
                else if(key == 'o'){
                    const float reduction = 0.5;
                    float scale_factor = 256.0 / (refS.height*reduction);
                    printf("Using scale factor %.2f\n", scale_factor);

                    cv::imwrite("output/fog_original.jpg", frame);
                    cv::Mat result = background_fog(frame, scale_factor);
                    cv::imwrite("output/fog_image.jpg", result);
                }
                else if(key == 't'){
                    special_effect(frame);
                }
                // Checking key press/ previous key press for saving images
                else if ( key == 's' )
                {
                    if( prev_key != 'g' && prev_key != 'h' && prev_key != 'a' && prev_key != 'b' && prev_key != 'f' && prev_key != 'p' ){
                        cv::imwrite("output/color_image.jpg", frame);
                    }
                    else if( prev_key == 'g' ){ // cvtColor gray scale
                        cv::imwrite("output/gray1_original.jpg", frame);
                        cv::Mat gray = gray1_filter(frame);
                        cv::imwrite("output/gray1_image.jpg", gray);
                    }
                    else if( prev_key == 'h' ){ // code implementation gray scale
                        cv::imwrite("output/gray2_original.jpg", frame);
                        cv::Mat gray = gray2_filter(frame);
                        cv::imwrite("output/gray2_image.jpg", gray);
                    }
                    else if( prev_key == 'a' ){ // sepia filter
                        cv::imwrite("output/sepia_original.jpg", frame);
                        cv::Mat sepia = sepia_filter(frame);
                        cv::imwrite("output/sepia_image.jpg", sepia);
                    }
                    else if( prev_key == 'b' ){ // blur filter
                        cv::imwrite("output/blur_original.jpg", frame);
                        cv::Mat dst;
                        blur5x5_2(frame, dst);
                        cv::imwrite("output/blur_image.jpg", dst);
                    }
                    else if(prev_key == 'f'){
                        frame = get_face_box(frame);
                        cv::imwrite("output/detected_face.jpg", frame);
                    }
                    else if(prev_key == 'p'){ // mirror filter
                        cv::imwrite("output/mirror_original.jpg", frame);
                        cv::Mat dst;
                        mirror_filter(frame, dst);
                        cv::imwrite("output/mirror_image.jpg", dst);
                    }
                }
                
                if ( key == 'g' || key == 'c' || key == 'h' || key == 'a' || key == 'b' || key == 'f' || key == 'p')
                    {
                        prev_key = key;
                    }    
        }

        delete capdev;
        return(0);
}