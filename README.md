# Iros-Challeng-Track-1
Team CVML code repo

# Phase 1
### 1. Preparing conda env
Assuming you have conda installed, let's prepare a conda env:
```
conda create -n drive python=3.10
pip install -r requirements.txt
```

### 2. Data Setting
```
# Setting INPUT_DATA="robosense_track1_realese_convert_temporal_5.json"
python convert_format.py --use-temporal --num-frames 5
# or just use "robosense_track1_realese_convert_temporal_5.json" on github

# Download FEW_SHOT_DATA="test_qa.json" data (https://huggingface.co/datasets/drive-bench/arena)
# This data is DriveBench Data
```
- Temporal Data : 5
- Few shot {k} : 10
- Model : Qwen-VL 32B

### 3. Deploy
We deploy the model using vLLM:
```
bash service.sh
```

### 4. Evaluate the baseline
Simply run:
```
bash inference_few_shot.sh
```

# Phase 2
### 0. Nuscenes Data
Download Nuscenes Trainval Dataset
```
Nuscenes
--- trainval
------ maps
------ samples
------ sweeps
------ v1.0-trainval
------ .v1.0-trainval_meta.txt
```

### 1. Data Processing
Process Data as following steps
```
# Extract
```

or, just download from link
