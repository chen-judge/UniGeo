# The name of experiment

output=/data1/chenjiaqi/Geo/06090_New2_lr5e-4_epoch20_batch5_solving_denoise_mask03

PYTHONPATH=$PYTHONPATH:./src \
python -m torch.distributed.launch \
    --nproc_per_node=$1 --master_port 20419 \
    src/pretrain.py \
        --distributed --multiGPU \
        --test_only \
        --train calculation_train \
        --valid calculation_val \
        --test calculation_val \
        --optim adamw \
        --warmup_ratio 0.1 \
        --lr 5e-4 \
        --epochs 20 \
        --batch_size 5 \
        --wordMaskRate 0.3 \
        --backbone 't5-base' \
        --output $output ${@:2} \
        --load $output/BEST \
        --num_beams 1 \
        --max_text_length 200 \
        --gen_max_length 200 \
        --num_workers 8 \
