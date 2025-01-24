# Exploring OpenCV Filters and Video Processing - Project1

## Overview
This project demonstrates various image and video filter techniques using OpenCV. It includes implementations of different filters, face detection, and effects on video streams, providing a comprehensive tutorial for understanding OpenCV's capabilities in computer vision.

---

## Features
- **Video Filters**:
  - Original Color filter
  - Gray Scale Filter (OpenCV's cvtColor)
  - Gray Scale Filter (Own interpretation)
  - Sepia Tone
  - Gaussian Blur (5x5 kernel)
  - Gaussian Blur (separable kernels)
  - Face detection with boxes
  - Mirror Effect
  - Portrait effect (background blur)
  - Fog effect 
  - Passing Circle

- **Image Filters (Applied to a single video frame)**:
  - Sobel X filter
  - Sobel Y filter
  - Magnitude
  - Blur Quantization
  - Median filter
  - Sketch Effect

---

## Project Structure

```
├── bin/
│   ├── #Executable binaries
├── images/                                 # Sample images for testing
│   ├── cathedral.jpeg
│   └── smile.jpg
├── include/                                # Includes for external libraries (if any)
├── output/                                 # Output folder for filter effected images
├── src/                                    # Source files
│   ├── DA2Network.hpp
│   ├── faceDetect.cpp
│   ├── faceDetect.h
│   ├── filters.cpp 
│   ├── filters.h
│   ├── haarcascade_frontalface_alt2.xml
│   ├── imgDisplay.cpp
│   ├── model_fp16.onnx
│   ├── timeBlur.cpp
│   └── vidDisplay.cpp
├── .gitignore                              # Git ignore file
├── makefile                                # Build configuration
```

---

## Tools used
- `OS`: MacOS
- `C++ Compiler`: Apple clang version 16.0.0
- `IDE`: Visual Studio code

---

## Dependencies
- OpenCV
- ONNX Runtime

**Note:** Update the dependency paths in the makefile after installation.

---

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

3. Compile the project:
   ```bash
   make
   ```

---

## Usage

### 1. Image Processing
Run the `imgDisplay` program to load and process a static image:
```bash
./bin/imgDisplay <image-path>
```
Press `q` to exit the program.

### 2. Video Processing
Run the `vidDisplay` program to process video streams with interactive filters or frame processing filters:
```bash
./bin/vidDisplay
```
**Keyboard Controls**:
- `c`: Original video
- `g`: Gray scale
- `h`: Alternate gray scale
- `a`: Sepia tone
- `b`: Gaussian blur
- `f`: Face detection
- `u`: Mirror effect
- `d`: Background blur
- `o`: Background fog
- `k`: Passing circle effect
- `x`: Sobel X filter
- `y`: Sobel Y filter
- `m`: Sobel magnitude filter
- `l`: Quantized blur
- `i`: Median filter
- `p`: Sketch effect

**Special Controls**:
- `s`: Save current frame
- `q`: Quit

More information about the internal implementation along with outputs of each of the above filters is included in **Project1_Report.pdf**

### 3. Time Blur
Run the `timeBlur` program to time two different blur implementations:
```bash
./bin/timeBlur
```

---

## Highlights
- The `filters.cpp` file includes multiple filter functions ranging 
    - Basic pixel modifications
    - Area pixel computations
    - Utilizing depth information to perform fog and portrait effects.
    - Face detection leverages Haar cascades for identifying faces in real-time video streams.

All these effects are utilized in vidDisplay.cpp.

---

## Contact
- **Name**: Saideep Arikontham
- **Email**: arikontham.s@northeastern,edu