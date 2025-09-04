# Iros-Challeng-Track-1
Team CVML code repo

# Preparing conda env
Assuming you have conda installed, let's prepare a conda env:
```
conda create -n drive python=3.10
pip install -r requirements.txt
```

# Data Setting
```
python convert_format.py --use-temporal --num-frames 5
```
- Temporal Data : 5
- Few shot {k} : 10
- Model : Qwen-VL 32B
