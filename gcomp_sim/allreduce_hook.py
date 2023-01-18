import torch
from .compressors.compressor import Compressor, NoneCompressor
from typing import Any, Tuple, Union, Iterator
import torch
import torch.distributed as dist


class SimCompressionState(object):
    def __init__(self, process_group: dist.ProcessGroup,
                 compressor: Compressor = NoneCompressor,
                 named_parameters: Iterator[Tuple[str, Any]] = None):
        self.process_group = process_group if process_group is not None else dist.group.WORLD
        self.param_offsets = {}
        self.total_size = 0
        self.step = 0
        self.layer_idx = 0
        self.layers_states = {}
        self.compressor = compressor
        if named_parameters:
            named_parameters = list(named_parameters)
            layer_id = len(named_parameters) - 1
            for name, p in named_parameters:
                if not p.requires_grad:
                    continue
                self.layers_states[layer_id] = {}
                self.layers_states[layer_id]["name"] = name
                self.layers_states[layer_id]["step"] = 0
                self.layers_states[layer_id]["layer_size"] = p.numel()
                layer_id -= 1


def _allreduce_fut(
        process_group: dist.ProcessGroup, tensor: torch.Tensor
) -> torch.futures.Future[torch.Tensor]:
    "Averages the input gradient tensor by allreduce and returns a future."
    group_to_use = process_group if process_group is not None else dist.group.WORLD

    # Apply the division first to avoid overflow, especially for FP16.
    tensor.div_(group_to_use.size())

    return (
        dist.all_reduce(tensor, group=group_to_use, async_op=True)
            .get_future()
            .then(lambda fut: fut.value()[0])
    )


def sim_compression_hook(
        state: SimCompressionState, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    if state.step >= 2:
        # This is because there is another internal optimization that rebuilds buckets at iteration 1 in DDP,
        # and this can conflict with any tensor indexation before the rebuild process.
        for tensor in bucket.gradients():
            if state.layer_idx not in state.layers_states:
                state.layers_states[state.layer_idx] = {"step": 0, "layer_id": state.layer_idx, "layer_size": tensor.numel(), "name": ""}
            else:
                if torch.distributed.get_rank() == 0 and state.layers_states[state.layer_idx]["layer_size"] != tensor.numel():
                    print("Missmatch of the stored layer sizes and provided by torch.distrbuted tensor."
                          " It might cause confusion in compressor if the compressor relies on the layers' states")
                    print(f"{bucket.index()}. Layer: state.layers_states[state.layer_idx]['name']. {state.layers_states[state.layer_idx]['layer_size']} vs {tensor.numel()}")
            layer_state = state.layers_states[state.layer_idx]
            layer_state["step"] = state.step
            if state.compressor:
                state.compressor.compress(tensor, layer_state)
            state.layer_idx += 1
    if bucket.is_last():
        # We can not rely on is_the_last_bucket_to_allreduce as it returns True after the first bucket
        state.step += 1
        state.layer_idx = 0
    return _allreduce_fut(state.process_group, bucket.buffer())
