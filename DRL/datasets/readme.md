## Datasets and Data preparation

Datasets and data preparation is an important component for understanding the deep representations, building computational models and then performing evaluation   through the abalation experiments undertaken in the hands on lab sessions.

## MNIST - Hand written digit dataset, a hello world to the Deep learning experiments.

where (“NIST” stands for National Institute of Standards and Technology while the “M”
stands for “modified” as the data has been preprocessed to reduce any burden on computer vision
processing and focus solely on the task of digit recognition) dataset is one of the most well studied
datasets in the computer vision and machine learning literature. The goal of this dataset is to correctly classify the handwritten digits 0 − 9.
In many cases, this dataset is a benchmark, a standard to which machine learning algorithms are ranked.

## CIFAR10 (standard benchmark dataset)- RGB images of dimension 32x32x3

where ("CIFAR" stands for Canadian Institute for Advanced Research) consists of 60,000 32×32×3 (RGB) images resulting in a feature vector dimensionality of 3072.
As the name suggests, CIFAR-10 consists of 10 classes, including: airplanes, automobiles, birds, cats, deer, dogs, frogs, horses, ships, and trucks.

### Environment setup ###

Tasks involved in setting up google-colab environment is composed of less number of steps compared to setting up of the environment on local computer.

Important command
from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)

Being a deep learning practitioner its important to understand  the right set of tools and packages. We are going to use colab. But the same scripts can be configured on the local computer/laptop by creating environments for individual study. Google colab is easier as it is flexible to use without any restrictions that will be imposed by the hardware associated with the computer/laptop.



This section details the programming language (python) along with the primary libraries that we will be using in Google Colab to study deep representation learning.

## Keras
To build and train our deep representation learning networks we’ll primarily be using the Keras library. Keras supports TensorFlow packages.
It is important to note the version of tensorflow and keras. Since the libraries are built in open source, we need to expect errors and Deprecated warnings appearing on the screen. The code currently working will be out of date after 6 months, depending on the updates happening with CUDA/CuDNN, tensorflow packages etc...

### cmd line installations of the required packages

## Installation of Tensorflow package
```!pip install tensorflow==1.13.1 ```

## Installation keras package
```!pip install keras==2.2.4 ```


## Instatllation of additional libraries
OpenCV, scikit-image, scikit-learn. Please use the commands, one at a time and seperate them into single lines before using it in the goolge colab.

```!pip install opencv-python ```
```!pip install scikit-image ```
```!pip install scikit-learn ```

## Summary

When it comes to configuring your deep learning development environment, you have a number of options. For the lab, we use google co-lab. If you would prefer to work from your local machine, that’s totally reasonable, but you will need to compile and install some dependencies first. If you are planning on using your
CUDA-compatible GPU on your local machine, a few extra install steps will be required as well. Download the appropriate version of CUDA compatible with the GPU of the local machine and further use compatible CuDNN package that goes hand in hand with the CUDA version.
