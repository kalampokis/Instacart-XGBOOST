wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
sudo apt-get update
sudo apt-get -y install cuda

sudo apt-get install python-pip

git config --global credential.helper gcloud.sh

pip install sklearn
pip install pandas
pip install Kaggle
pip install xgboost 

mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/

cd Instacart-XGBOOST

python instacart-xgboost-gridsearch.py