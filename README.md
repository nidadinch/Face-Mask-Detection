# Face-Mask-Detection
Image Processing Project Face Mask Detection via Keras/Tensorflow, Python, OpenCV &amp; MobileNet


## About The Project

It is a project for Akdeniz University Computer Science Department - Image Processing Lesson.

### INTRODUCTION TO PROBLEM
Medical masks are being used for a long time in history. Due to the pandemic situation we are in the middle of, medical masks are having much more importance day by day. The virus we are trying to avoid from, cannot be reach a human body itself; it can only reach from the eyes and nose or mouth. Medical masks help us in that point.
There is a 100% accurate solution to end this pandemic with stopping the whole daily life for a 15 days, however the life must go on, people has to work, firms must reach their goals, manufacturing cannot stop. So, the masks must be used. The main problem with that is, not every human being is aware of the situation and they are not giving enough importance to masks.
Our system that we made is checking human faces to determine if they are using masks or not. It can be replaced as a control mechanism in the enterance of big buildings, hospitals, work places and any place that can be a possibility of a non-controllable crowd. It can be improved with an alarm system which gives a warning message to the security or it can also block the entering system and lock the doors for that person which does not have a mask.
The difficulty with the development is image’s properties (pose of the head, location of the head, mask’s color and type) must be appropriate to check the mask. We fixed the problem with using a large set of data to train our program.

### SURVEY
The methods that are can be seperated as “face detection + edge detection” and “training the program + deep learning”.
In the first case program finds the face with “haar features” and viola-jones algorithm. Then looks for a mask. It does searching for a mask with comparing the pixel values on the matrices. The problem with that is, we cannot make the perfect environment every time. There can be errors due to light, exposure, blurring, auto-focus error etc. and the matrices’ values cannot be comparable or can give wrong results.
In the second case which we used, we basically followed the steps which are: Collect a dataset (faces with mask and without mask) Load the dataset and train the program (with selecting which one includes a mask and which one does not). Serialize model to disk (save the brain of the program) Detect faces and extract them ROI. Apply our “modal” to the detected faces and check for a mask. Show the result.

### METHOD DESCRIPTION
Face Mask Detection via Keras/Tensorflow, Python, OpenCV & MobileNet
Our method to develop that mask detection can be seperated as “phase1: train mask detector” and “phase 2: apply mask detector”.

* Phase 1 - Step 1 : Load the dataset (659 images with mask, 680 images with images)
* Phase 1 - Step 2: Preprocess the data, send it to MobileNet and then do max-pooling
[![Figure](https://github.com/nidadinch/Face-Mask-Detection/blob/main/Images/Figure.png)](https://github.com/nidadinch/Face-Mask-Detection/blob/main/Images/Figure.png)
* Phase 1 - Step 2 : Train using model
* Phase 1 - Step 3: Serialize mask model to disk
[![Figure1](https://github.com/nidadinch/Face-Mask-Detection/blob/main/Images/Figure2.png)](https://github.com/nidadinch/Face-Mask-Detection/blob/main/Images/Figure2.png)
* Phase 2 - Step 1: Load mask model from disk (from the phase 1 step 3)
* Phase 2 - Step 2 : Detect faces in image or in video
[![Figure2](https://github.com/nidadinch/Face-Mask-Detection/blob/main/Images/Figure1.png)](https://github.com/nidadinch/Face-Mask-Detection/blob/main/Images/Figure1.png)
* Phase 2 - Step 3 : Extract each face ROI
* Phase 2 - Step 4 : Apply mask model to each face ROI to check the mask
* Phase 2 - Step 5 : Show the result

[![Figure4](https://github.com/nidadinch/Face-Mask-Detection/blob/main/Images/without_mask.png)](https://github.com/nidadinch/Face-Mask-Detection/blob/main/Images/without_mask.png)
[![Figure3](https://github.com/nidadinch/Face-Mask-Detection/blob/main/Images/with_mask.png)](https://github.com/nidadinch/Face-Mask-Detection/blob/main/Images/with_mask.png)



### DATASET
For first method (face mask detection using keras/tensorflow, python, opencv and mobilenet) we have used 3835 images for dataset.
the dataset has 3835 images of faces. all images collected from following sources:
- kaggle datasets - rmfd dataset
- bing search apı
and dataset divided to two categories:
- without_mask : 1916 images - with_mask : 1919 images

### EXPERIMENTATION
We have used all images for train the model. After training we archieved 0.9909 accurancy using mobileNet & max-pooling method.
When we use noisy images, accurancy dropped to 0.9689.

## Built With

* Python
* OpenCV

## Getting Started

Train :

* ```sh
   python3 train_mask_detector.py

   ```
Live Video:
* ```sh
   pyhton3 detect_mask_video.py

   ```


## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Referances

[1]- https://www.pyimagesearch.com/2020/05/04/covid-19-face-mask-detector-with-opencv-keras-tensorflow-and-deep-learning/

[2]- https://www.mygreatlearning.com/blog/real-time-face-detection/

[3]- https://drive.google.com/drive/folders/1XDte2DL2Mf_hw4NsmGst7QtYoU7sMBVG

[4]- https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset

## License

Distributed under the MIT License. See `LICENSE` for more information.


## Contact

Nida Dinç - niddinc@gmail.com

Project Link: [https://github.com/nidadinch/Face-Mask-Detection](https://github.com/nidadinch/Face-Mask-Detection)
