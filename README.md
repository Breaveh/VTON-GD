# A 3D VIRTUAL TRY-ON METHOD WITH GLOBAL-LOCAL ALIGNMENT AND DIFFUSION MODEL


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

### 4. Testing DM
```sh
cd DM
python run.py -p test -c config/inpainting_MPV.json
```

### 5. Testing RDG
```sh
python test.py --model RDG --name RDG --dataroot path/to/data --warproot path/to/warp --datalist test_pairs --results_dir results
```

### 6. Getting colored point cloud and Remeshing

(Note: since the back-side person images are unavailable, in `rgbd2pcd.py` we provide a fast face inpainting function that produces the mirrored back-side image after a fashion. One may need manually inpaint other back-side texture areas to achieve better visual quality.)

```sh
python rgbd2pcd.py
```

Now you should get the point cloud file prepared for remeshing under `results/aligned/pcd/test_pairs/*.ply`. [MeshLab](https://www.meshlab.net/) can be used to remesh the predicted point cloud, with two simple steps below:

- Normal Estimation: Open MeshLab and load the point cloud file, and then go to Filters --> Normals, Curvatures and Orientation --> Compute normals for point sets

- Possion Remeshing: Go to Filters --> Remeshing, Simplification and Reconstruction --> Surface Reconstruction: Screen Possion (set reconstruction depth = 9)


## Training on MPV3D Dataset

With the pre-processed MPV3D dataset, you can train the model from scratch by folllowing the three steps below:

### 1. Train SGN

```sh
python train.py --model SGN --name SGN --dataroot path/to/MPV3D/data --datalist train_pairs --checkpoints_dir path/for/saving/model
```

then run the command below to obtain the `--warproot` (here refers to the `--results_dir`) which is necessary for the other two modules:
```sh
python test.py --model SGN --name SGN --dataroot path/to/MPV3D/data --datalist train_pairs --checkpoints_dir path/to/saved/MTMmodel --results_dir path/for/saving/MTM/results
```

### 2. Train GLA

```sh
python train.py --model GLA --name GLA --dataroot path/to/MPV3D/data --datalist train_pairs --checkpoints_dir path/for/saving/model
```

then run the command below to obtain the `--warproot` (here refers to the `--results_dir`) which is necessary for the other two modules:
```sh
python test.py --model GLA --name GLA --dataroot path/to/MPV3D/data --datalist train_pairs --checkpoints_dir path/to/saved/MTMmodel --results_dir path/for/saving/MTM/results
```

### 3. Train P

```sh
python train.py --model P --name P --dataroot path/to/MPV3D/data --warproot path/to/warp --datalist train_pairs --checkpoints_dir path/for/saving/model
```

### 4. Train DM

```sh
cd DM
python run.py -p train -c config/inpainting_MPV.json
```
### 5. Train RDG

```sh
python train.py --model RDG --name RDG --dataroot path/to/MPV3D/data --warproot path/to/warp --datalist train_pairs --checkpoints_dir path/for/saving/model
```

(See options/base_options.py and options/train_options.py for more training options.)

## Failure Case
Backpacks can cause self occlusion and unconventional posture of the human body. Our model currently cannot solve this problem very well. This issue is currently an unresolved challenge in the Try On field, and our future work will focus on it.
(left: Human, right: Failed results).
![Image text](https://github.com/Breaveh/VTON-GD/blob/main/img/example1.png) ![Image text](https://github.com/Breaveh/VTON-GD/blob/main/img/example1tryon.png)

![Image text](https://github.com/Breaveh/VTON-GD/blob/main/img/example2.png) ![Image text](https://github.com/Breaveh/VTON-GD/blob/main/img/example2tryon.png)

![Image text](https://github.com/Breaveh/VTON-GD/blob/main/img/example3.png) ![Image text](https://github.com/Breaveh/VTON-GD/blob/main/img/example3tryon.png)

## License
The use of this code is RESTRICTED to non-commercial research and educational purposes.

