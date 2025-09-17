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
Process qa data with nuscenes devkit
```
python qa_2_nuscenes.py \
  --input robosense_track1_phase2_convert_temporal_5.json \
  --output final_nuscene.json \
  --nusc_root /nuscenes/trainval \
  --nusc_version v1.0-trainval
```

or, just download from link
https://drive.google.com/file/d/1XQq9ioAV9lBcYK_eG6Ev3dWCdg39DsRe/view?usp=drive_link

### 2. Split QA per category
For categoty specific methods, we first split qa.
```
python split.py
```

### 3. Deploy
We deploy the model using vLLM:
```
bash service.sh
```

### 4. Evaluate mathod per category
for test you set --max_history_frames 0
Simply run:
```
# For perception-MCQs
bash inference_perception_mcq.sh

# For other perceptions
bash inference_other_perception.sh

# For prediction
bash inference_prediction.sh

# For corruption and planning
bash inference_planning_and_corruption.sh

```

### 5. Merge QA
Finally, merging qa for final results
```
python merge.py
```

# Final Results
Phase 1 and phase2 results is on here:
```
phase1_final_results.json
phase2_final_results.json
```
