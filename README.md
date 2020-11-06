# XTYT-CNN

Here, we provide a PyTorch implementation of our method proposed in the paper

"Spatio-Temporal Deep Learning-BasedUndersampling Artefact Reduction for 2D Radial Cine MRI WithLimited Training Data"

by A. Kofler, M. Dewey, T. Schaeffter, W. Wald, and C. Kolbitsch

see also: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8793147 or https://arxiv.org/abs/1904.01574

Unlike in the paper, this implementation allows the processing of complex-valued 2D cine MR images. This is done (as it's mostly the case) by stacking real- and imaginary parts of the images as channels. Further, the method implements the "change of perspective" on the data by switching to xt- and yt-domain within the network and thus allowing it to be ingrated as a building block in iterative or cascaded networks. In the original paper, on the other hand, the xt- and yt-slices are processed offline and the image series are obtained by reassembling the processed slices as a post-processing step.

Further, we provide the model (i.e. the weights) of the CNN E3 C4 K64 following the notation in the paper.

If you find the code useful or if you use for your work, please cite the following:

@article{kofler2019spatio, title={Spatio-temporal deep learning-based undersampling artefact reduction for 2D radial cine MRI with limited training data}, author={Kofler, Andreas and Dewey, Marc and Schaeffter, Tobias and Wald, Christian and Kolbitsch, Christoph}, journal={IEEE transactions on medical imaging},
