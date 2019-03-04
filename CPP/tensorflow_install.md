# Install GPU support

## Install nvidia driver

```
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-get update
sudo apt-get install nvidia-384
```


## Install CUDA

Download `cuda-repo-ubuntu1604-9-0-local_9.0.176-1_amd64.deb` from  https://developer.nvidia.com/cuda-90-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1604&target_type=deblocal

```
sudo dpkg -i cuda-repo-ubuntu1604-9-0-local_9.0.176-1_amd64.deb
sudo apt-key add /var/cuda-repo-9-0-local/7fa2af80.pub
sudo apt-get update
sudo apt-get install cuda
```

Make sure the symlink is correct. `readlink -f /usr/local/cuda`
> /usr/local/cuda-9.0


## Install cuDNN
Download `libcudnn7_7.3.1.20-1+cuda9.0_amd64.deb` `libcudnn7-dev_7.3.1.20-1+cuda9.0_amd64.deb` `libcudnn7-doc_7.3.1.20-1+cuda9.0_amd64.deb` from  https://developer.nvidia.com/rdp/cudnn-archive (click: Download cuDNN v7.3.1 (Sept 28, 2018), for CUDA 9.0 )

```
sudo dpkg -i libcudnn7_7.3.1.20-1+cuda9.0_amd64.deb
sudo dpkg -i libcudnn7-dev_7.3.1.20-1+cuda9.0_amd64.deb
sudo dpkg -i libcudnn7-doc_7.3.1.20-1+cuda9.0_amd64.deb
sudo cp /usr/include/cudnn.h /usr/local/cuda/include
sudo cp /usr/lib/x86_64-linux-gnu/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
```

Add environment variable

```
echo 'export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"' >> ~/.bashrc
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
source ~/.bashrc
```


## Install opencv-3.4.2

Install opencv-3.4.2 according to https://docs.opencv.org/3.4.2/d7/d9f/tutorial_linux_install.html

## Dependency

```
sudo apt-get install build-essential
sudo apt-get install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
sudo apt-get install python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev

```


## Generate and compile

```
cd <working_directory>
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git

# change code to 3.4.2
cd opencv_contrib && git checkout 3.4.2 && cd ..
cd opencv && git checkout 3.4.2 && cd ..

cd opencv
mkdir build
cd build



# <PATH_TO> set it to your opencv_contrib root path
# <YOUR_CUDA_ARCH> set it to your CUDA_ARCH, you can find it from  http://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/

cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=/usr/local/opencv-3.4.2/  \
-D OPENCV_EXTRA_MODULES_PATH=<PATH_TO>/opencv_contrib/modules \
-D WITH_TBB=ON \
-D WITH_V4L=ON \
-D WITH_GTK=ON \
-D WITH_OPENGL=ON \
-D WITH_CUDA=ON \
-D WITH_CUBLAS=ON \
-D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 \
-D CUDA_ARCH_BIN=<YOUR_CUDA_ARCH>  \
-D CUDA_ARCH_PTX="" \
-D CUDA_FAST_MATH=ON \
-D BUILD_EXAMPLES=OFF \
-D INSTALL_PYTHON_EXAMPLES=OFF \
-D INSTALL_C_EXAMPLES=OFF \
..

# For build with viz :
- `sudo apt-get isntall libvtk5-dev`
- cmake with `-D WITH_VTK=ON`

make -j8

sudo make install

sudo ln -s /usr/local/opencv-3.4.2 /usr/loca/opencv

```


## Install Bazel
Bazel is the build tool of tensorflow. Details of Bazel installation is as follows. Up to date information can be found [Here](https://docs.bazel.build/versions/master/install-ubuntu.html).

- Step 1: Install required packages `sudo apt-get install pkg-config zip g++ zlib1g-dev unzip python `
- Step 2: Download Bazel `bazel-0.18.1-installer-linux-x86_64.sh` from https://github.com/bazelbuild/bazel/releases
- Step 3: Run the installer

```
chmod +x bazel-0.18.1-installer-linux-x86_64.sh
./bazel-0.18.1-installer-linux-x86_64.sh --user
```

The `--user` flag installs Bazel to the $HOME/bin directory on your system and sets the .bazelrc path to $HOME/.bazelrc

- Step 4: Set up your environment

If you ran the Bazel installer with the `--user` flag, the Bazel will be installed in your `$HOME/bin directory`. To add this directory to the default search path, run the following shell command,

```
export PATH="$PATH:$HOME/bin" 
```


## Buid Tensorflow

- `git clone --recursive https://github.com/tensorflow/tensorflow`
- `cd tensorflow`  `git checkout r1.9`
- `./configure`

```
Do you wish to build TensorFlow with jemalloc as malloc support? [Y/n]: n   
Do you wish to build TensorFlow with Google Cloud Platform support? [Y/n]: n
Do you wish to build TensorFlow with Hadoop File System support? [Y/n]: n
Do you wish to build TensorFlow with Amazon S3 File System support? [Y/n]: n
Do you wish to build TensorFlow with Apache Kafka Platform support? [Y/n]: n
Do you wish to build TensorFlow with XLA JIT support? [y/N]: n
Do you wish to build TensorFlow with GDR support? [y/N]: n
Do you wish to build TensorFlow with VERBS support? [y/N]: n
Do you wish to build TensorFlow with OpenCL SYCL support? [y/N]: n
Do you wish to build TensorFlow with CUDA support? [y/N]: y
Please specify the CUDA SDK version you want to use. [Leave empty to default to CUDA 9.0]: 9.0
Please specify the location where CUDA 9.0 toolkit is installed. Refer to README.md for more details. [Default is /usr/local/cuda]:
Please specify the cuDNN version you want to use. [Leave empty to default to cuDNN 7.0]: 7.3.1
Please specify the location where cuDNN 7 library is installed. Refer to README.md for more details. [Default is /usr/local/cuda]:
Do you wish to build TensorFlow with TensorRT support? [y/N]: n
Please specify the NCCL version you want to use. [Leave empty to default to NCCL 1.3]
Do you want to use clang as CUDA compiler? [y/N] n
Please specify which gcc should be used by nvcc as the host compiler. [Default is /usr/bin/gcc]:
Do you wish to build TensorFlow with MPI support? [y/N]: n
Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native]:
Would you like to interactively configure ./WORKSPACE for Android builds? [y/N]: n
```

- Build tensorflow c++ library  
```
bazel build --config=monolithic //tensorflow:libtensorflow_cc.so
```

- Install tensorflow to `/usr/local/tensorflow`

```
sudo mkdir /usr/local/tensorflow
sudo mkdir /usr/local/tensorflow/include
# <TENSORFLOW_CACHE_HASH> is where bazel tmp store the download files when build tensorflow
sudo cp -r ~/.cache/bazel/_bazel_<USR_NAME>/<TENSORFLOW_CACHE_HASH>/external/eigen_archive/Eigen /usr/local/tensorflow/include/
sudo cp -r ~/.cache/bazel/_bazel_<USR_NAME>/<TENSORFLOW_CACHE_HASH>/external/eigen_archive/unsupported /usr/local/tensorflow/include/
sudo cp -r ~/.cache/bazel/_bazel_<USR_NAME>/<TENSORFLOW_CACHE_HASH>/external/protobuf_archive/src/google /usr/local/tensorflow/include/
sudo cp -r bazel-genfiles/tensorflow /usr/local/tensorflow/include/
sudo cp -r tensorflow/cc /usr/local/tensorflow/include/tensorflow
sudo cp -r tensorflow/core /usr/local/tensorflow/include/tensorflow
sudo mkdir /usr/local/tensorflow/include/third_party
sudo cp -r third_party/eigen3 /usr/local/tensorflow/include/third_party/

sudo mkdir /usr/local/tensorflow/lib
sudo cp bazel-bin/tensorflow/libtensorflow_*.so /usr/local/tensorflow/lib

```
