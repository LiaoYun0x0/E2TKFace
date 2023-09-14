# E2TKFace
Accurate 3D face reconstruction from a single 2D image remains a challenging task in computer vision because existing neural networks lack rotation robustness. In this paper, we propose a novel approach called E2TKFace, which combines a group equivariant convolutional neural network architecture with a window attention mechanism for 3D face alignment and reconstruction. E2TKFace uses a UV position map to represent the 3D face, allowing the prediction of over 50,000 points using a 256-size UV map. Experimental results demonstrate the effectiveness and robustness of E2TKFace in handling challenging and complex facial poses.
## Requirements:
    python 3.6.9
    opencv-python 4.1
    pillow 6.1
    pyrender 0.1.32
    pytorch 1.1.0
    scikit-image 0.15.0
    scikit-learn 0.21.3
    scipy 1.3.1
    tensorboard 2.0.0
    torchvision 0.3.0

## Getting started
Please refer to [face3d](https://github.com/YadiraF/face3d/blob/master/examples/Data/BFM/readme.md) to prepare BFM data. And move the generated files in Out/
 to data/Out/
(Thanks for their opensource code. The codes in faceutil are mainly from face3d)

Download databases from [3DDFA](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm). Put the raw dataset in data/images (e.g. 
data/images/AFLW2000)

Run processor.py to generate UV position maps. I recommend you to use the following instructions:
```cmd
python processor.py -i=data/images/AFLW2000 -o=data/images/AFLW2000-crop -f=True -v=True --isOldKpt=True
python processor.py -i=data/images/300W_LP -o=data/images/300W_LP-crop --thread=16

```
It takes about 2-4 hours depending on your working machine.

Trian PRNet:
```cmd
python torchrun.py -train=True -test=False --batchSize=16 -td=data/images/300W_LP-crop -vd=data/images/AFLW2000 --numWorker=1

```
If you have more than 128 GB RAM, you can use
```cmd
python torchrun.py -train=True -test=False --batchSize=16 -td=data/images/300W_LP-crop -vd=data/images/AFLW2000 --isPreRead=True --numWorker=8
```

For multi-gpus, for example 4 gpus, you can set:
```cmd
--gpu=4 --visibleDevice=0,1,2,3
```


Evaluation, use your own  model path, for example:
```cmd
python torchrun.py -train=False -test=True -pd=data/images/AFLW2000 --loadModelPath=savedmodel/temp_best_model/2019-11-18-12-34-19/best.pth

```


