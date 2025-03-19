
python evaluate.py --size 50 --task gender --version base --seed 68
python evaluate.py --size 18 --task gender --version base --seed 68
python evaluate.py --size 50 --task gender --version quant --seed 68
python evaluate.py --size 18 --task gender --version quant --seed 68
python fineTune.py --task gender --size 50 --seed 80
python fineTune.py --task gender --size 18 --seed 80
python quantize.py --size 50 --task gender --seed 80
python quantize.py --size 18 --task gender --seed 80
python evaluate.py --size 50 --task gender --version base --seed 80
python evaluate.py --size 18 --task gender --version base --seed 80
python evaluate.py --size 50 --task gender --version quant --seed 80
python evaluate.py --size 18 --task gender --version quant --seed 80
python fineTune.py --task gender --size 50 --seed 708
python fineTune.py --task gender --size 18 --seed 708
python quantize.py --size 50 --task gender --seed 708
python quantize.py --size 18 --task gender --seed 708
python evaluate.py --size 50 --task gender --version base --seed 708
python evaluate.py --size 18 --task gender --version base --seed 708
python evaluate.py --size 50 --task gender --version quant --seed 708
python evaluate.py --size 18 --task gender --version quant --seed 708
@REM Future me you need to change quantize and evaluate 