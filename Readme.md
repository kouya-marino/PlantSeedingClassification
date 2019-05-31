##Make three folders name as data,output,segmented_image


## Download datset from following kaggle link and extart them in data folder 
https://www.kaggle.com/c/plant-seedlings-classification/data
 


there are two python code(plant_new_2.py and seg.py)

for classification we used transfer learning(using vgg16) and tranfer the knowledge to 12 class we have. 

python plant_new_2.py


After we will get 3 file in output folder a image of performance measure of our model on validation data , a image of confussion matix on validation data and a test csv file resulting the test data prediction.

we got an validation accuracy of ~97%

python seg.py

its show how the segemented of image take place and show one image_segmentation from each class in segmented_image folder.
 
  
