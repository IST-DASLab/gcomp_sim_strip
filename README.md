# Gcomp_sim
Gcomp_sim is a pytorch-based framework for gradient compression simulation.
The framework can be used either with Horovod or with `torch.nn.parallel.DistributedDataParallel`.

## Installation
Requires torch 1.12.
``` bash
python setup.py install
```

## Usage

### Torch.DDP
```python
import gcomp_sim
import torch
model = ...

model = torch.nn.parallel.DistributedDataParallel(
    model,
    device_ids=[args.local_rank],
    output_device=args.local_rank,
    find_unused_parameters=False,
)
compressor = gcomp_sim.NoneCompressor()
comp_state = gcomp_sim.SimCompressionState(torch.distributed.group.WORLD, compressor=compressor)
model.register_comm_hook(comp_state, gcomp_sim.sim_compression_hook)
```

## Extension
In order to create their own compressors user has to inherit `gcomp_sim.Compressor` class and implement `compress_(self, grad, state)` method.
The method accepts parameter `grad` as tensor which has to be changed in-place as is would be compressed-decompressed (simulated) and parameter `state`,
a dictionary that contains compression state of the corresponding layer/module. For all compressors state contains "step" key, corresponding to the current step of training.
Also `gcomp_sim.Compressor` has parameter `enable_error_correction` allowing to use error feedback and `warmup_steps` postponing the first compression in the training.
