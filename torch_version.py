import torch
print(torch.version.cuda)

import transformers
import deepspeed
print("transformers version", transformers.__version__)
print("deepspeed version", deepspeed.__version__)
import trl
print("trl version", trl.__version__)