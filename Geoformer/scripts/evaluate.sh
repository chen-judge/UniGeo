# inference
output=snap/test

PYTHONPATH=$PYTHONPATH:./src \
python -m torch.distributed.launch \
    --nproc_per_node=$1 --master_port 2098 \
    src/geo.py \
        --distributed --multiGPU \
        --test_only \
        --train calculation_train \
        --valid calculation_val \
        --test calculation_test,proving_test \
        --optim adamw \
        --warmup_ratio 0.1 \
        --lr 1e-3 \
        --epochs 100 \
        --num_workers 4 \
        --backbone 't5-base' \
        --output $output ${@:2} \
        --load $output/BEST \
        --num_beams 10 \
        --batch_size 10 \
        --max_text_length 200 \
        --gen_max_length 40 \