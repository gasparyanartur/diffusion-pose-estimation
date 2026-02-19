# Setup

Requirements:
* Python 3.10
* CUDA >= 12.1

``` 
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt   
```

## Install mm-pose
```
git clone https://github.com/open-mmlab/mmpose.git external/mmpose
cd external/mmpose
git checkout v1.3.2
pip install -e .
cd ../..
```

# Development

## Dataset

https://mmpose.readthedocs.io/en/latest/dataset_zoo/2d_body_keypoint.html#crowdpose

