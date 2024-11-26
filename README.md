# Setup on lambda labs
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh # takes a long time to scrol, see if auto-accept?
/home/ubuntu/miniconda3/bin/conda init

# Create env
conda create --name zero python=3.10
conda activate zero
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
pip install deepspeed transformers
pip install datasets tokenizers accelerate huggingface-hub trl

# Train DPO with logs
python trainlog2.py --experiment_name exp1 --report_to none

# DS1
CUDA_VISIBLE_DEVICES=0,1 deepspeed --num_gpus 2 deepspeed1.py --experiment_name ds_dev1 --report_to none

# DS4 -- this works
deepspeed --num_gpus 8 ds4.py --experiment_name ds_gpt2 --report_to none


# DS5 -- this works
deepspeed --num_gpus 8 ds5.py --experiment_name llamaNoZeRO --report_to none --deepspeed ds_llama.json

deepspeed ds5.py --experiment_name llamaNoZeRO --report_to none --deepspeed deepspeed_cfg/NoZeRO.json


accelerate launch --deepspeed_plugin=deepspeed_cfg/ZeRO2OptCPU.json --num_processes 4 ds5.py --experiment_name ZeRO2OptCPU_acc --report_to none


# Acc
accelerate launch --config_file=deep.yaml acc.py
[rank0]: Traceback (most recent call last):
[rank0]:   File "/home/ubuntu/Finetune/acc.py", line 315, in <module>
[rank0]:     main()
[rank0]:   File "/home/ubuntu/Finetune/acc.py", line 304, in main
[rank0]:     trainer.train()
[rank0]:   File "/home/ubuntu/miniconda3/envs/zero/lib/python3.10/site-packages/transformers/trainer.py", line 2123, in train
[rank0]:     return inner_training_loop(
[rank0]:   File "/home/ubuntu/miniconda3/envs/zero/lib/python3.10/site-packages/transformers/trainer.py", line 2480, in _inner_training_loop
[rank0]:     with context():
[rank0]:   File "/home/ubuntu/miniconda3/envs/zero/lib/python3.10/contextlib.py", line 135, in __enter__
[rank0]:     return next(self.gen)
[rank0]:   File "/home/ubuntu/miniconda3/envs/zero/lib/python3.10/site-packages/accelerate/accelerator.py", line 973, in no_sync
[rank0]:     with context():
[rank0]:   File "/home/ubuntu/miniconda3/envs/zero/lib/python3.10/contextlib.py", line 135, in __enter__
[rank0]:     return next(self.gen)
[rank0]:   File "/home/ubuntu/miniconda3/envs/zero/lib/python3.10/site-packages/deepspeed/runtime/engine.py", line 1995, in no_sync
[rank0]:     assert not self.zero_optimization_partition_gradients(), \
[rank0]: AssertionError: no_sync context manager is incompatible with gradient partitioning logic of ZeRO stage 2