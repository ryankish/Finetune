# Train DPO with logs
python trainlog2.py --experiment_name exp1 --report_to none

# DS1
CUDA_VISIBLE_DEVICES=0,1 deepspeed --num_gpus 2 deepspeed1.py --experiment_name ds_dev1 --report_to none

# DS4 -- this works
deepspeed --num_gpus 8 ds4.py --experiment_name ds_gpt2 --report_to none


# DS5 -- this works
deepspeed --num_gpus 8 ds5.py --experiment_name llamaNoZeRO --report_to none --deepspeed ds_llama.json
