## Project Overview

This repository hosts code used in second project on Convolutional Neural Networks (CNN) project in the Deep Learning Nanodegree. In addition to meeting the project specifications, I also added on a few things. 

Given an image of a dog, the algorithm will identify an estimate of the canine’s breed.  If supplied an image of a human, it will identify the resembling dog breed.  

## Getting started
Open the `dog_app.ipynb` on a GPU enabled Python notebook environment. The notebook consists of further technical details. Run the notebook to classify your own dog images.

A flask webapp has been created that detects dogs or humans.

## libraries used in webapp
see requirements.txt

![alt text](https://github.com/louisbrown1/Project-4/blob/master/Screenshot%202023-09-25%205.31.14%20PM.png?raw=true)

##  app folder structure
      |-- app
            |-- templates
                    |-- start.html
                    |-- results.html
            |-- run.py
            |-- weights.best.VGG19.hdf5
            |-- extract_bottleneck_features.py

## Instructions:
Run your web app: python run.py
