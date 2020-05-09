# 3D_voxels

Keras implementation of the [VoxelGrid](https://arxiv.org/abs/1711.06396) paper

Classification of 3D models of modelnet, by converting the data to its voxel representation. 

Package requirement: Python3.6, keras, tensorflow, numpy, matplotlib, h5py

# Training:

1. Download the aligned dataset from https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip.
2. Put all traning h5 files under Prepdata folder, all testing h5 files under Prepdata_test folder, then run train_cls.py.</br>
</br>
Accuracy rate will be 82%. 
