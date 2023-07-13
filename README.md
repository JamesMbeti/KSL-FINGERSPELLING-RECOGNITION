# KSL-FINGERSPELLING-RECOGNITION
https://drive.google.com/file/d/1gzWecIMdBkWFQYJBd87OqNgs_XCe0LHU/view?usp=drive_link

##### Contributors: [James Mbeti](https://github.com/JamesMbeti) , [Taylor Musa](https://github.com/ojutaylor), [Brian Kisilu](https://github.com/Bkitainge), [Sharon Kimani](https://github.com/Sharonkimani), [Nelson Odhiambo](https://github.com/mandelaGit), [Vivian Adhiambo](https://github.com/vivianOpondo)

## Table of contents 
- [Business Understanding](#business-understanding)
- [Data preparation](#data-preparation)
- [Modeling](#modeling)
- [Evaluation](#evaluations)

---

## 1. Business Understanding
### Overview
Fingerspelling is a technique that makes use of hand formations to represent words and letters. Using fingerspelling, one can communicate information such as phone numbers, names, and even addresses. However, hearing impaired individuals find it difficult to merge the current technological advancements, such as, smartphones with fingerspelling because they happen to do it a lot faster than their devices. Therefore, there is need to bridge the gap between fingerspelling and typing on smartphones.

Fingerspelling has grown to become one of the most crucial manual communication systems in the world. For deaf and individuals with hearing impairment issues, fingerspelling is one of the ways that they can communicate with those around them. However, few people are able to discern figerspelling and this leads to communication breakdowns. The use of machine learning technology can help to merge fingerspelling with exact letters and symbols to help enhance communication.

### Problem Statement

The deaf and hearing impaired community faces significant communication barriers with the rest of the society. This is because sign language is not widely understood by everyone else around them, and this can lead to difficulties in communication or communication breakdowns. To address this issue, this project aims to develop a Convolutional Neural Network (CNN) model specifically designed for fingerspelling recognition, allowing for accurate identification of individual letters and complete words in different sign languages. By improving the recognition of fingerspelling gestures, the model seeks to enhance communication accessibility for individuals who are deaf or hard of hearing, promoting inclusivity and fostering effective communication with the broader society.

### Objectives

#### Main objective:

* The main objective is to create an innovative machine learning model that acts as a vital bridge, connecting the deaf and mute community with the wider society by translating fingerspelling images to text.

#### Specific objectives:

* Develop a Convolutional Neural Network (CNN) model specifically designed for fingerspelling recognition in different sign languages.
* Train the model using a large dataset of fingerspelling gestures in various sign languages, ensuring accuracy and reliability in recognizing individual letters and complete words.
* Conduct extensive testing and evaluation to assess the model's performance and accuracy in recognizing fingerspelling gestures across different sign languages.
* Deploy the model.

## 2. Data Understanding
> kaggle dataset
* The dataset is publicly available on Kaggle 
* The signs provided are based on the Kenyan Sign Language letter database which is made up of 24 classes of letters with the exclusion of J and Z. The two letters have been excluded because they require the use of motion.


> Raw dataset




------
## 3. Data Preparation
Within our data preparation phase, we performed the following tasks:
* Clean Data
* Checking Duplicates
* image processing
* Feature Engineering 


------
## 4. EDA
The following analysis was performed on the data:
* Previewing the images in the data
* univariate Analysis on the labels


------
## 5. Modeling 
The models include;
* Dense neural network
* Convolution neural network

-------
## 5. Evaluation 
Our success metrics is accuracy. The model that had the highest accuracy for the validation set was chosen to be the best model and thus used to predict the fingerspelling images.

## 6. Conclusions





---

## 7. Recommendations
