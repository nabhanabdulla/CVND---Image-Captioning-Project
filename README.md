# CVND---Image-Captioning-Project

Deep learning research has flourished since AlexNet in 2012. Today, variants of ResNet and EfficientNet algorithms have attained results on par to humans in image classification tasks. But, humans don't just detect objects but also can comprehend the complex relationships between them in the real world.

Image captioning is about understanding relations between objects in a scene and coming up with a description of it in natural language. Two sets of algorithms work subsequently to make image captioning possible - namely Convolutional Neural Network(CNN) and Recurrent Neural Network(RNN). The former known in the deep learning world for its ability to work wonders with images spits out a summary of the image which the latter utilises to forge awesome text description of the scene

# Todo
- [X] Add project description

- [ ] Add applications of image captioning

- [ ] Explain the helper functions in detail
  - [ ] data_loader
  - [ ] vocabulary
  - [ ] model
  
- [ ] Explain project in detail

# Contents
1. Project Overview
2. Applications
3. Helper functions
4. Project Walkthrough
5. Instructions to set up locally

# Instructions  
1. Clone this repo: https://github.com/cocodataset/cocoapi  
```
git clone https://github.com/cocodataset/cocoapi.git  
```

2. Setup the coco API (also described in the readme [here](https://github.com/cocodataset/cocoapi)) 
```
cd cocoapi/PythonAPI  
make  
cd ..
```

3. Download some specific data from here: http://cocodataset.org/#download (described below)

* Under **Annotations**, download:
  * **2014 Train/Val annotations [241MB]** (extract captions_train2014.json and captions_val2014.json, and place at locations cocoapi/annotations/captions_train2014.json and cocoapi/annotations/captions_val2014.json, respectively)  
  * **2014 Testing Image info [1MB]** (extract image_info_test2014.json and place at location cocoapi/annotations/image_info_test2014.json)

* Under **Images**, download:
  * **2014 Train images [83K/13GB]** (extract the train2014 folder and place at location cocoapi/images/train2014/)
  * **2014 Val images [41K/6GB]** (extract the val2014 folder and place at location cocoapi/images/val2014/)
  * **2014 Test images [41K/6GB]** (extract the test2014 folder and place at location cocoapi/images/test2014/)

4. The project is structured as a series of Jupyter notebooks that are designed to be completed in sequential order (`0_Dataset.ipynb, 1_Preliminaries.ipynb, 2_Training.ipynb, 3_Inference.ipynb`).
