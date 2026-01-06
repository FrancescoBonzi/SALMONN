# How to Train SALMONN 13B on LibriSpeech on Tamia

Download the code and install packages (optional: create env)
```
git clone https://github.com/FrancescoBonzi/SALMONN
git checkout feature/mutor
cd SALMONN
pip install -r requirements.txt
```

Download and prepare LibriSpeech dataset
```
python recipes/librispeech/prepare_librispeech.py
```

Download the pretrained models
```
bash recipes/download_pretrained.sh
```

Start training on TamIA using sbatch
```
sbatch run.sh
```
