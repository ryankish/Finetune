import torch
import torch.distributed as dist

dist.init_process_group(backend="nccl", init_method="env://", world_size=1, rank=0)
print(f"Rank {dist.get_rank()} initialized successfully.")
dist.barrier()
print("Barrier passed successfully.")
import torch.distributed as dist
dist.destroy_process_group()
