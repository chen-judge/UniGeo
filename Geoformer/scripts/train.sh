# The name of experiment
output=snap/test

PYTHONPATH=$PYTHONPATH:./src \
python -m torch.distributed.launch \
    --nproc_per_node=$1 --master_port 2132 \
    src/geo.py \
        --distributed --multiGPU \
        --train calculation_train,proving_train \
        --valid calculation_val,proving_val \
        --test calculation_test,proving_test \
        --optim adamw \
        --warmup_ratio 0.1 \
        --clip_grad_norm 5 \
        --lr 2e-4 \
        --batch_size 10 \
        --epochs 100 \
        --num_workers 8 \
        --backbone 't5-base' \
        --output $output ${@:2} \
        --num_beams 1 \
        --max_text_length 200 \
        --gen_max_length 40 \
        --load snap/pretrained