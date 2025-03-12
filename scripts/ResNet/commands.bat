python fineTune.py --task gender --size 50 --seed 42
python fineTune.py --task gender --size 18 --seed 42
python quantize.py --size 50 --task gender --seed 42
python quantize.py --size 18 --task gender --seed 42
python evaluate.py --size 50 --task gender --version base --seed 42
python evaluate.py --size 18 --task gender --version base --seed 42
python evaluate.py --size 50 --task gender --version quant --seed 42
python evaluate.py --size 18 --task gender --version quant --seed 42
@REM Future me you need to change quantize and evaluate 