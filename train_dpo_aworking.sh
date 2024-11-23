python train_dpo_aworking.py \
    --output_dir="./dpo_output" \
    --num_train_epochs=3 \
    --per_device_train_batch_size=4 \
    --gradient_accumulation_steps=4 \
    --learning_rate=5e-5 \
    --warmup_ratio=0.1 \
    --logging_steps=10 \
    --save_steps=100 \
    --eval_steps=100