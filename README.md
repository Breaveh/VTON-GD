# A 3D VIRTUAL TRY-ON METHOD WITH GLOBAL-LOCAL ALIGNMENT AND DIFFUSION MODEL

## Requirements
```python >= 3.8.0, pytorch == 1.6.0, torchvision == 0.7.0```

## Data Preparation

### MPV3D Dataset
Please downloading the [MPV3D Dataset](https://github.com/fyviezhao/M3D-VTON) and run the following script to preprocess the data:
```sh
python util/data_preprocessing.py --MPV3D_root path/to/MPV3D/dataset
```

### Custom Data

If you want to process your own data, please refer to [this](https://github.com/fyviezhao/M3D-VTON) to process the data and place the data in the corresponding folder.

## Running Inference
We provide demo inputs under the `mpv3d_example` folder.

With inputs from the `mpv3d_example` folder, the easiest way to get start is to use the [pretrained models](https://figshare.com/s/fad809619d2f9ac666fc) and sequentially run the four steps below:

### 1. Testing SGN
```sh
python test.py --model SGN --name SGN --dataroot path/to/data --datalist test_pairs --results_dir results
```

### 2. Testing GLA 
```sh
python test.py --model GLA --name GLA --dataroot path/to/data --datalist test_pairs --results_dir results
```  

### 3. Testing P
```sh
python test.py --model P --name P --dataroot path/to/data --warproot path/to/warp --datalist test_pairs --results_dir results
```

### 4. Testing RDG
```sh
python test.py --model RDG --name RDG --dataroot path/to/data --warproot path/to/warp --datalist test_pairs --results_dir results
```

### 5. Testing DM
```sh
cd DM
python run.py -p train -c config/inpainting_MPV.json
```

### 6. Getting colored point cloud and Remeshing

(Note: since the back-side person images are unavailable, in `rgbd2pcd.py` we provide a fast face inpainting function that produces the mirrored back-side image after a fashion. One may need manually inpaint other back-side texture areas to achieve better visual quality.)

```sh
python rgbd2pcd.py
```

Now you should get the point cloud file prepared for remeshing under `results/aligned/pcd/test_pairs/*.ply`. [MeshLab](https://www.meshlab.net/) can be used to remesh the predicted point cloud, with two simple steps below:

- Normal Estimation: Open MeshLab and load the point cloud file, and then go to Filters --> Normals, Curvatures and Orientation --> Compute normals for point sets

- Possion Remeshing: Go to Filters --> Remeshing, Simplification and Reconstruction --> Surface Reconstruction: Screen Possion (set reconstruction depth = 9)

Now the final 3D try-on result should be obtained:

![Try-on Result](/assets/meshlab_snapshot.png "Try-on Result")

## Training on MPV3D Dataset

With the pre-processed MPV3D dataset, you can train the model from scratch by folllowing the three steps below:

### 1. Train MTM module

```sh
python train.py --model MTM --name MTM --dataroot path/to/MPV3D/data --datalist train_pairs --checkpoints_dir path/for/saving/model
```

then run the command below to obtain the `--warproot` (here refers to the `--results_dir`) which is necessary for the other two modules:
```sh
python test.py --model MTM --name MTM --dataroot path/to/MPV3D/data --datalist train_pairs --checkpoints_dir path/to/saved/MTMmodel --results_dir path/for/saving/MTM/results
```

### 2. Train DRM module

```sh
python train.py --model DRM --name DRM --dataroot path/to/MPV3D/data --warproot path/to/MTM/warp/cloth --datalist train_pairs --checkpoints_dir path/for/saving/model
```

### 3. Train TFM module

```sh
python train.py --model TFM --name TFM --dataroot path/to/MPV3D/data --warproot path/to/MTM/warp/cloth --datalist train_pairs --checkpoints_dir path/for/saving/model
```

(See options/base_options.py and options/train_options.py for more training options.)

## License
The use of this code and the MPV3D dataset is RESTRICTED to non-commercial research and educational purposes.

## Citation
If our code is helpful to your research, please cite:
```
@InProceedings{M3D-VTON,
    author    = {Zhao, Fuwei and Xie, Zhenyu and Kampffmeyer, Michael and Dong, Haoye and Han, Songfang and Zheng, Tianxiang and Zhang, Tao and Liang, Xiaodan},
    title     = {M3D-VTON: A Monocular-to-3D Virtual Try-On Network},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {13239-13249}
}
```

