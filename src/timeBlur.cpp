/*
  Bruce A. Maxwell
  Spring 2024
  CS 5330 Computer Vision

  Example of how to time an image processing task.

  Program takes a path to an image on the command line
*/

#include <cstdio> // a bunch of standard C/C++ functions like printf, scanf
#include <cstring> // C/C++ functions for working with strings
#include <cmath>
#include <sys/time.h>
#include "opencv2/opencv.hpp"

// prototypes for the functions to test
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

int blur5x5_2( cv::Mat &src, cv::Mat &dst ){
    // separable filter
    int kernel[5] = {1, 2, 4, 2, 1};

    src.copyTo( dst );

    // Horizontal
    for (int i=0; i<src.rows; i++){
        cv::Vec3b *ptr = dst.ptr<cv::Vec3b>(i);

        for (int j=2; j<src.cols-2; j++){
            int blue = 0, green = 0, red = 0;

            blue = ptr[j-2][0]*kernel[0] + ptr[j-1][0]*kernel[1] + ptr[j][0]*kernel[2] + ptr[j+1][0]*kernel[3] + ptr[j+2][0]*kernel[4];
            green = ptr[j-2][1]*kernel[0] + ptr[j-1][1]*kernel[1] + ptr[j][1]*kernel[2] + ptr[j+1][1]*kernel[3] + ptr[j+2][1]*kernel[4];
            red = ptr[j-2][2]*kernel[0] + ptr[j-1][2]*kernel[1] + ptr[j][2]*kernel[2] + ptr[j+1][2]*kernel[3] + ptr[j+2][2]*kernel[4];

            ptr[j][0] = blue/10;
            ptr[j][1] = green/10;
            ptr[j][2] = red/10;
        }
    }

    // Vertical
    for (int i=2; i<src.rows-2; i++){
        for (int j=0; j<src.cols; j++){
            int blue = 0, green = 0, red = 0;

            blue = dst.at<cv::Vec3b>(i - 2, j)[0] * kernel[0] + dst.at<cv::Vec3b>(i - 1, j)[0] * kernel[1] + dst.at<cv::Vec3b>(i, j)[0] * kernel[2] + dst.at<cv::Vec3b>(i + 1, j)[0] * kernel[3] + dst.at<cv::Vec3b>(i + 2, j)[0] * kernel[4];
            green = dst.at<cv::Vec3b>(i - 2, j)[1] * kernel[0] + dst.at<cv::Vec3b>(i - 1, j)[1] * kernel[1] + dst.at<cv::Vec3b>(i, j)[1] * kernel[2] + dst.at<cv::Vec3b>(i + 1, j)[1] * kernel[3] + dst.at<cv::Vec3b>(i + 2, j)[1] * kernel[4];
            red = dst.at<cv::Vec3b>(i - 2, j)[2] * kernel[0] + dst.at<cv::Vec3b>(i - 1, j)[2] * kernel[1] + dst.at<cv::Vec3b>(i, j)[2] * kernel[2] + dst.at<cv::Vec3b>(i + 1, j)[2] * kernel[3] + dst.at<cv::Vec3b>(i + 2, j)[2] * kernel[4];

            dst.at<cv::Vec3b>(i, j)[0] = blue/10;
            dst.at<cv::Vec3b>(i, j)[1] = green/10;
            dst.at<cv::Vec3b>(i, j)[2] = red/10;
        }
    }

    return 0;
}

// returns a double which gives time in seconds
double getTime() {
  struct timeval cur;

  gettimeofday( &cur, NULL );
  return( cur.tv_sec + cur.tv_usec / 1000000.0 );
}
  

// argc is # of command line parameters (including program name), argv is the array of strings
// This executable is expecting the name of an image on the command line.

int main(int argc, char *argv[]) {  // main function, execution starts here
  cv::Mat src; // define a Mat data type (matrix/image), allocates a header, image data is null
  cv::Mat dst; // cv::Mat to hold the output of the process
  char filename[256]; // a string for the filename

  // usage: checking if the user provided a filename
  if(argc < 2) {
    printf("Usage %s <image filename>\n", argv[0]);
    exit(-1);
  }
  strcpy(filename, argv[1]); // copying 2nd command line argument to filename variable

  // read the image
  src = cv::imread(filename); // allocating the image data
  // test if the read was successful
  if(src.data == NULL) {  // src.data is the reference to the image data
    printf("Unable to read image %s\n", filename);
    exit(-1);
  }

  const int Ntimes = 10;
	
  //////////////////////////////
  // set up the timing for version 1
   printf("Blur 1 results:\n");
  double startTime = getTime();

  // execute the file on the original image a couple of times
  for(int i=0;i<Ntimes;i++) {
    blur5x5_1( src, dst ); 
  }

  // end the timing
  double endTime = getTime();

  // compute the time per image
  double difference = (endTime - startTime) / Ntimes;

  // print the results
  printf("- Time per image (1): %.4lf seconds\n", difference );

  ////////////////////////////
  // set up the timing for version 2

  printf("Blur 2 results:\n");
  startTime = getTime();

  // execute the file on the original image a couple of times
  for(int i=0;i<Ntimes;i++) {
    blur5x5_2( src, dst );
  }

  // end the timing
  endTime = getTime();

  // compute the time per image
  difference = (endTime - startTime) / Ntimes;

  // print the results
  printf("- Time per image (2): %.4lf seconds\n", difference );
  
  // terminate the program
  printf("\nTerminating\n");

  return(0);
}
