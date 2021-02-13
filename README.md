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
Mask detection is a new problem that we face recently so, there is not that much different methods to determine if there is a mask in a picture or in a video. The methods to detect the mask can be simply separated as “face detection + edge detection” and “training the program + deep learning”.

In the first case we can talk about the history of face detection which is a long-time problem when compared to mask detection. [5]

We can say that recently face detection algorithms are having much more importance and so, engineers work on that to improve it and to get better results. The main problem they are trying to figure out is to detect even if the environment is complex such as cluttered backgrounds and low quality images. Some of the algorithms that used are still too computationally expensive to be apply for a real time processing. However, this can be fixed with coming improvements in computer hardware technology.

We can analyze methods separated as “feature-based” and “image-based”.

Feature based methods can be used in real-time systems where color and motion is available. The main problem with that is, these methods cannot always provide visual cues to focus attention due to exhaustive multi resolution window scanning cannot always be preferable. In that case, the common approach to fix that problem is “skin color detection”.

Image-based approaches are the most powerful techniques to process gray-scale images. Sung and Poggio and Rowley et al. Developed an algorithm on that topic and that algorithms is still can be used because it is still comparable with recent common algorithms.The high computational cost can be decreased with avoiding multi resolution window scanning with combining these two approaches with using visual clues like skin color when we are trying to find the face.

To conclude, detecting a face is still a hard problem to solve, considering the changes in faces over time like facial hair, glass usage, etc.
In the second case which we used to determine if there is a mask or not, we basically followed the steps which are: collect a dataset with and without mask, load the dataset and train the program with selecting which ones have mask and which ones do not, serialize model to disk, detect faces and extract them ROI, apply our model to detect faces and check for a mask and finally show the result.It is much more robust solution because in that case we did not have to deal with the problems like cluttered background sor low quality images, we also did not deal with collecting a dataset to use visual clues about the skin color. To sum up, we did not focus on the face but the mask.
The other specific reason that we picked that one is to learn the recent technological approaches to problems when compared to the old ones.

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



### DATASET
For face mask detection using keras/tensorflow, python, OpenCV and mobilenet we have used 3835 images for dataset. All images are extracted from Kaggle datasets and RMFD dataset, Bing Search API. And all images are real. Images from all three sources are equated. The ratio of masked faces to unmasked faces indicates that the data set is balanced.

If our dataset and model require a lot of training, that is, if the model has too many parameters to adjust, then we have to use a larger dataset for training, which is our case.

We needed to divide our dataset into two parts; train dataset & test dataset. And first of all, we have decided to use 60% of dataset to training the model. And other 40% of the dataset to use for testing the model. After that, we have decided to retrain & retest our model by using 80% of the dataset for training the model & 20% for the test. We want to see difference of two training’s accurancy, recall & precision.
The dataset has 3835 images of faces. All images collected from following sources: [3] [4]

- Kaggle datasets - Rmfd dataset
- Bing search API
And each image provides only 1 face. Dataset is balanced & divided into two categories:
- without_mask : 1916 images - with_mask : 1919 images

### EXPERIMENTATION
First of all, we have used 60% of the dataset to training the model. And we tested the model using 40% of the dataset.. And we archived 0.85 accuracy using mobileNet & max-pooling method. After that, we have used 80% of the dataset for training the model. After training we archieved 0.9909 accurancy using mobileNet & max-pooling method.As a result of training, 99% precision and 99% recall in the segmentation of face detection
When we use noisy images, accurancy dropped to 0.8689.
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

[5]- https://www.cin.ufpe.br/~rps/Artigos/Face%20Detection%20- %20%20A%20Survey.pdf
## License

Distributed under the MIT License. See `LICENSE` for more information.


## Contact

Nida Dinç - niddinc@gmail.com

Project Link: [https://github.com/nidadinch/Face-Mask-Detection](https://github.com/nidadinch/Face-Mask-Detection)
