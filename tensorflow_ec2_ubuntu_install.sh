# This is taken almost completely from http://ramhiser.com/2016/01/05/installing-tensorflow-on-an-aws-ec2-instance-with-gpu-support/
# Not a single new thing here, just an easier way to find this 

sudo apt-get update
sudo apt-get upgrade
sudo apt-get install -y build-essential git python-pip libfreetype6-dev libxft-dev libncurses-dev libopenblas-dev gfortran python-matplotlib libblas-dev liblapack-dev libatlas-base-dev python-dev python-pydot linux-headers-generic linux-image-extra-virtual unzip python-numpy swig python-pandas python-sklearn unzip wget pkg-config zip g++ zlib1g-dev
sudo pip install -U pip

wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1410/x86_64/cuda-repo-ubuntu1410_7.0-28_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1410_7.0-28_amd64.deb
rm cuda-repo-ubuntu1410_7.0-28_amd64.deb
sudo apt-get update
sudo apt-get install -y cuda

curl -O http://developer.download.nvidia.com/compute/redist/cudnn/v2/cudnn-6.5-linux-x64-v2.tgz
tar -zxf cudnn-6.5-linux-x64-v2.tgz && rm cudnn-6.5-linux-x64-v2.tgz
sudo cp -R cudnn-6.5-linux-x64-v2/lib* /usr/local/cuda/lib64/
sudo cp cudnn-6.5-linux-x64-v2/cudnn.h /usr/local/cuda/include/

sudo reboot

export CUDA_HOME=/usr/local/cuda
export CUDA_ROOT=/usr/local/cuda
export PATH=$PATH:$CUDA_ROOT/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_ROOT/lib64

sudo add-apt-repository -y ppa:webupd8team/java
sudo apt-get update
# Hack to silently agree license agreement
echo debconf shared/accepted-oracle-license-v1-1 select true | sudo debconf-set-selections
echo debconf shared/accepted-oracle-license-v1-1 seen true | sudo debconf-set-selections
sudo apt-get install -y oracle-java8-installer

# Make sue bazel verison is right
sudo apt-get install pkg-config zip g++ zlib1g-dev
wget https://github.com/bazelbuild/bazel/releases/download/0.3.2/bazel-0.3.2-installer-linux-x86_64.sh
chmod +x bazel-0.3.2-installer-linux-x86_64.sh
./bazel-0.3.2-installer-linux-x86_64.sh --user
rm bazel-0.3.2-installer-linux-x86_64.sh
PATH=$PATH:/home/ubuntu/bin

git clone --recurse-submodules https://github.com/tensorflow/tensorflow
cd tensorflow
# If want CUDA, accept defaults everywhere except for:
# CUDA support - Y
# cuDNN version - 6.5
# Cuda compute capabilities - 3.0
# For setup without CUDA, just accept all defaults
TF_UNOFFICIAL_SETTING=1 ./configure

# without CUDA
bazel build -c opt //tensorflow/cc:tutorials_example_trainer
bazel build -c opt //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
sudo pip install --upgrade /tmp/tensorflow_pkg/tensorflow-0.12.0rc0-cp27-none-linux_x86_64.whl

# with CUDA
bazel build -c opt --config=cuda //tensorflow/cc:tutorials_example_trainer
bazel build -c opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
sudo pip install --upgrade /tmp/tensorflow_pkg/tensorflow-0.12.0rc0-cp27-none-linux_x86_64.whl