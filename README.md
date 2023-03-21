# Semantic Segmentation of Satellite Images to Identify Clouds

## Problem Description
According to NASA, only about 30% of the Earth’s land is completely free of clouds. This cloud cover presents a major challenge for satellite imagery analysis since clouds often obscure the land features we want to see. Such a hindrance can impede any number of tasks, from monitoring deforestation to estimating energy consumption. The goal of the project is to create an algorithm that can identify which pixels in a 512x512 pixel image are clouds and which are not clouds. For the uninitiated, the project involves semantic segmentation, or the assignment of individual pixels from an image to particular groups. I completed this project alongside two graduate student classmates as part of the Applied Machine Learning course from UC Berkeley’s Master’s of Information and Data Science program.

## Data
Microsoft’s Planetary Computer stores 10,986 “chips” from the European Space Agency’s Sentinel 2 satellites, comprising roughly 28 GB of data in total. Every chip contains multiple images, each capturing a certain bandwidth of light, for instance from the blue, green, red, and near infrared spectrums. These different bands can be combined to create a human-readable image. Additionally, each chip also came with a mask highlighting which pixels were cloud pixels and where were not.
