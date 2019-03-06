# Traffic light detection based on light-weight CNN

---
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

This repo is part of Udacity final project. I utilize Tensorflow Object Detection API based on SSD-MobileNetV2/SSD-InceptionV2 architecture for traffic light detection. Note that SSD-MobileNetV2 cannot be compatible with old version tensorflow, I use Tensorflow 1.13 for training.

## Training dataset 

[dataset](https://github.com/alex-lechner/Traffic-Light-Classification#training
)


## Configure Tensorflow Object Detection API


Tensorflow object detection API installation on [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)


## Modify the config file of the pretrained model

Mainly, the following items should be adapted to the new task.

* num_classes
* max_detections_per_class (origin is 100, maybe change into a lower number)
* fine_tune_checkpoint
* train_input_reader
* eval_input_reader
* batch_norm_trainable: true (maybe delete it to avoid the error)


## Google Colab for training the network

Please check the [notebook](https://github.com/karlTUM/udacity_traffic_light_detection/blob/master/google_colab_work/work/train_tf_traffic_light_detection.ipynb) for training the network.


## Results:

For training the simulation dataset, I train around 20000 iterations. Below is the Average Precision on the validation dataset during training.

![alt text](./pic/sim_eval_ssd_mobileV2.png)
*AP under different levels of IoU on the validation dataset during training*

![alt text](./pic/simu_example1.png)
![alt text](./pic/simu_example2.png)
![alt text](./pic/simu_example3.png)

*Detection results on simulated dataset*

---
## Real dataset

### SSD-InceptionV2

For real dataset, the image size is about 1306x1000, but the object area is very small around 20x50. In that case, I first try to increase the network detection power and  utilize SSD-InceptionV2 model based on resized 512x512 images. 

The training codes on Google Colab can be found in notebook.

Below are the results:

![alt text](./pic/real_example1.png)
![alt text](./pic/real_example2.png)
![alt text](./pic/real_example3.png)
![alt text](./pic/real_example4.png)
![alt text](./pic/real_example5.png)
![alt text](./pic/real_example6.png)
![alt text](./pic/real_example7.png)
![alt text](./pic/real_example8.png)

### SSD-MobileNetV2

The results of light-weight SSD-MobileNetV2 on real dataset is in the [notebook](https://github.com/karlTUM/udacity_traffic_light_detection/blob/master/google_colab_work/work/reference_ssd_mobilenetV2_real_data.ipynb). 





