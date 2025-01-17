/*
Saideep Arikontham
January 2025
CS 5330 OpenCV tutorial
*/

#include <cstdio> // gives printf
#include <cstring> // gives strcpy
#include <opencv2/opencv.hpp> // openCV


int main(int argc, char *argv[]) {
    cv::Mat src;
    char filename[256];
    int key_pressed;
    // check for a command line argument
    if(argc < 2 ) {
        printf("usage: %s <image filename>\n", argv[0]); // argv[0] is the program name
        exit(-1);
    }

    strncpy(filename, argv[1], 255); // safe strcpy
    src = cv::imread( filename ); // by default, returns image as 8-bit BGR image (if it's color), use IMREAD_UNCHANGED to keep the original data format

    if( src.data == NULL) { // no data, no image
        printf("error: unable to read image %s\n", filename);
        exit(-2);
    }

    cv::imshow( filename, src ); // display the original image

    // Entering loop to wait for a key press - "q" to quit
    while(1){
        key_pressed = cv::waitKey(0); // returns ASCII for pressed key
        if(key_pressed == 113 || key_pressed == 81){ // ASCII for 'q' (113) and 'Q' (81)
            printf("key pressed: %c, terminating\n", static_cast<char>(key_pressed));
            exit(0); // exit the loop and terminate the program
        } 
        else{
            printf("key pressed: %c, continuing\n", static_cast<char>(key_pressed));
        }
    }
}