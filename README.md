# Text-localization-in-Images
![GX1E6](https://user-images.githubusercontent.com/74713842/116364398-ca847e80-a821-11eb-9b20-c9d4928c3bdf.png)


Fully Convolutional Neural Network to localize the text in a image using EAST Research paper
![EAST_paper_main_blocks](https://user-images.githubusercontent.com/74713842/116355405-6eb4f800-a817-11eb-8f49-b5c12cb04da3.PNG)

# UPDATE in TEXT Localization <h1> 
  
  ### I have uploaded updated code which is localizing even text with some rotation, file name is modified_detector.py <h3>
  
  
  ### Here is the original image<h3>
  ![car_rot](https://user-images.githubusercontent.com/74713842/116805208-7ee41480-ab42-11eb-8a75-abbb8974a709.jpg)
  
  ### previous output <h3>
![previous_car_rot](https://user-images.githubusercontent.com/74713842/116805210-85728c00-ab42-11eb-91ec-5df34e6ca264.jpg)
  
  ### Modified output <h3>
![modified_Car_rot](https://user-images.githubusercontent.com/74713842/116805213-8a374000-ab42-11eb-8c1f-612e4596c92f.jpg)

  
  

# Installation guide <h1> 

# In order to use bbox_text.py or noplate_ocr.py file you need to create the Environment <h1> 


### 1.use conda create -n [env name] python==3.6.9 <h3> 
### 2.conda activate [env name]<h3> 
### 3.pip install -r requirements.txt  <h3> 
### 4.download the frozen graph at this URL: https://drive.google.com/file/d/1lwHGhEwRo0c8aBrTgXtvBJ_rFSDEo-at/view?usp=sharing<h3> 
### 5.Place the frozen graph in the root directory <h3> 
### 6.place some sample images in the root directory<h3> 
### 7.navigate to root folder and execute bbox_text.py<h3> 



# In order to see the architecture of the model you can see frozen_east_text_detection.png in github <h1> 
  
  ![Capture1](https://user-images.githubusercontent.com/74713842/116363283-978dbb00-a820-11eb-8e22-9bf1edf14768.PNG)

# Here is one of the sample output <h1> 
  ![sample](https://user-images.githubusercontent.com/74713842/116363510-d459b200-a820-11eb-92da-2c460fb79bfb.png)
![sample_text](https://user-images.githubusercontent.com/74713842/116363529-d91e6600-a820-11eb-8ee2-bf2628d0c05e.jpg)


# Number Plate Recognition <h1> 
### navigate to root folder and execute noplate_ocr.py<h3> 
# Here are some of the sample Ouputs <h1> 
  
  ![board_text](https://user-images.githubusercontent.com/74713842/116520612-94fa9680-a8f0-11eb-981e-402037f46d13.jpg)
![car5_text](https://user-images.githubusercontent.com/74713842/116520663-a2178580-a8f0-11eb-8cf7-b13db0ba8f37.jpg)
