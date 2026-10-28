# Batch Facial Expression Prediction

## 1. Description

The Batch Facial Expression Prediction System predicts facial expressions from multiple images and counts the number of images classified as Happy, Sad, Angry, and Neutral.

**Project Workflow:**

Image Collection → Data Processing → Face Detection → Expression Prediction → Count Calculation → Result Analysis

## 2. Methodology

1. **Image Collection:** Select multiple facial images from the computer.
2. **Data Processing:** Convert images to grayscale and resize the detected faces to the required dimensions.
3. **Face Detection:** Detect faces in each image using OpenCV's Haar Cascade classifier.
4. **Expression Prediction:** Use a trained machine learning model to predict facial expressions.
5. **Count Calculation:** Count the predictions for each expression category.
6. **Result Analysis:** Display individual predictions and the final expression counts.

## 3. Input / Output

### Input

Multiple facial images are provided to the system for expression prediction.

![Input - Angry](images/angry.jpg)

![Input - Happy](images/happy.jpg)

![Input - Neutral](images/neutral.jpg)

![Input - Sad](images/sad.jpg)

### Output

The program displays the predicted expression for each image, saves the images with predicted labels, and calculates the total count for each category.

![Output - Angry](images/angry_prediction.jpg)

![Output - Happy](images/happy_prediction.jpg)

![Output - Neutral](images/neutral_prediction.jpg)

![Output - Sad](images/sad_prediction.jpg)

**Example Output:**

- Happy = 1
- Sad = 1
- Angry = 1
- Neutral = 1



## GitHub Repository

Repository: [Batch Facial Expression Prediction](https://github.com/Ch6785/batch-facial-expression-prediction)

## Related Project

Project 1: [Facial Expression Recognition](https://github.com/Ch6785/facial-expression-recognition)
