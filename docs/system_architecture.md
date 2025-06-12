# System Architecture Overview

## Introduction
The Sign Language Recognition System is designed to convert sign language gestures into text and speech. The system consists of three main components: a Raspberry Pi with a camera and audio output, a server for processing the data, and an Android application for user interaction.

## Components

### 1. Raspberry Pi
- **Camera Module**: Captures real-time video feed of sign language gestures.
- **Audio Output**: Converts recognized text into speech and plays it through connected speakers.
- **Configuration**: Contains settings for server URL and camera parameters.

### 2. Server
- **Flask Application**: Acts as the main server to handle incoming requests from the Raspberry Pi.
- **Sign Language Model**: A pre-trained machine learning model that recognizes sign language gestures from images.
- **API Endpoints**: 
  - Receives image data from the Raspberry Pi.
  - Processes the images and returns the recognized text to the Raspberry Pi.

### 3. Android Application
- **User Interface**: Provides a simple interface for users to interact with the system.
- **API Client**: Manages communication with the server, sending image data and receiving recognized text.
- **Display**: Shows the recognized text on the screen and triggers audio output.

## Workflow
1. The Raspberry Pi captures an image of a sign language gesture using the camera module.
2. The image is sent to the server via an API call.
3. The server processes the image using the sign language model and returns the recognized text.
4. The Raspberry Pi receives the text and uses the audio output module to convert it to speech.
5. The Android application displays the recognized text and allows users to interact with the system.

## Conclusion
This architecture enables real-time recognition of sign language gestures, facilitating communication for the hearing and speech impaired. The integration of hardware and software components ensures a seamless user experience.