# The name of experiment
dir=NewPretrain
name=test

output=snap/Geo/$dir/$name

PYTHONPATH=$PYTHONPATH:./src \
python -m torch.distributed.launch \
    --nproc_per_node=$1 --master_port 20417 \
    src/pretrain.py \
        --distributed --multiGPU \
        --train calculation_train \
        --valid calculation_val \
        --test calculation_test \
        --optim adamw \
        --warmup_ratio 0.1 \
        --lr 5e-4 \
        --epochs 1 \
        --batch_size 5 \
        --wordMaskRate 0.3 \
        --backbone 't5-base' \
        --output $output ${@:2} \
        --num_beams 1 \
        --max_text_length 200 \
        --gen_max_length 200 \
        --num_workers 8 \

