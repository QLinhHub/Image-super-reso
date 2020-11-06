# Image-super-reso
Super-resolution is the task of constructing a higher-resolution image from a lower-resolution image. While this task has traditionally been approched with non-linear methodes such as bilinear and bicubic upsampling, neural networks offer an opportunity for significant improvements. 

Inspired from the Residual Dense Network for Image Restoration, the paper publiced at 23 Jan 2020 of Yulun Zhang, Yapeng Tian, Yu Kong, Bineng Zhong, and Yun Fu, Fellow, IEEE, i have implemented the Residual Dense Network for image super-resolution (RDN for image SR).
Link to the paper: [https://arxiv.org/pdf/1812.10477.pdf

I draw the dataset from DIV2K, a dataset of 1000 high-resolution images with diverse subjects. Due to the constraints on GPU rescources, i used a subsample of 100 images, extracted to the size 256x256, so i got 2870 images for train and 780 images for validation. To generate lower-resolution images from higher-resolution images, i used max pooling method.

For the result i got the PSNR for the validation set is 32db after 54 epochs (using early stopping).
I have used my model to evaluate some classic images(lena, pepper...) but not the images from DIV2K, and i got 28db for PNSR. This score indicate that my best model can beat the bicubic upsampling for this images super-resolution task which just 23db. The result images is found in *savepath* folder.

