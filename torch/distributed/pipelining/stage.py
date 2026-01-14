# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
import logging
import operator
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, cast, Optional, Union

import torch
import torch.distributed as dist
import torch.fx as fx
import torch.nn as nn
from torch._subclasses.fake_tensor import FakeTensor
from torch.distributed._composable.replicate_with_fsdp import replicate, ReplicateModule
from torch.distributed._mesh_layout import _MeshLayout
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FSDPModule, fully_shard
from torch.distributed.pipelining._utils import (
    _DTensorMeta,
    _make_tensor_from_meta,
    _MeshCache,
    _StageBackwardMeta,
    _StageForwardMeta,
    _StageMeta,
    _TensorMeta,
    extract_tensor_meta,
    extract_tensor_metas,
    flatten_args,
    InferenceMode,
    normalize_to_tuple,
    PipeInfo,
    PipeliningMetadataError,
    TensorMeta,
    to_local_if_dtensor,
    validate_metadata,
    validate_static_dtensor_grad_correspondence,
    validate_tensors_metadata,
)
from torch.distributed.tensor import DTensor
from torch.fx.node import Argument, map_aggregate
from torch.nn.parallel import DistributedDataParallel

from ._backward import stage_backward, stage_backward_input, stage_backward_weight
from ._debug import map_debug_info


__all__ = [
    "PipelineStage",
    "build_stage",
]

logger = logging.getLogger(__name__)


def _normalize_model_output_as_tuple(output: Any) -> tuple[Any]:
    """[Note: pipeline model output type]

    The output of the model passed to pipelining can be any type, controlled by the user.

    However, there are 2 API surfaces that complicate this.
    (1) the outputs of intermediate stages are passed via Send/Recv ops to subsequent stages. The implicit assumption
    is that each element of the outputs is a tensor.  Otherwise, Send/Recv would not be supported.  The exception
    is the last layer of the model, which can output anything any which won't be communicated via Send/Recv.
    (2) the outputs of the last layer of the model are returned to the user, or, passed to the loss function.
    The loss function can be written in any way, such that its inputs match the outputs of the model.

    It would be convenient if we could strictly type the output signature of the pipeline stage wrapping the model,
    but we do not want to impose an unnecessary constraint on user provided models.

    Currently, we let user provided models return either a Tensor or a tuple of Tensors from each stage. Due to
    torch.export tracing, compiled models may also return a list instead of a Tuple, which we will normalize back to a
    tuple for consistency.

    TODO: should we be stricter about asserting that stage modules (intermediate and output) all return only Tensor
    values?
    """
    if type(output) is list:
        # HACK: this is a hacky workaround for the fact that export creates
        # output in list format
        output = tuple(output)

    # Unify output form to tuple for easy correspondence with
    # `act_send_info`
    output_tuple = output if type(output) is tuple else (output,)
    return output_tuple


class _RecvInfo:
    """
    Represents an input tensor to a pipeline stage.

    This class handles both:
    1. Received activations from a previous stage (is_root_arg=False)
    2. Root-level model inputs provided by the user (is_root_arg=True)

    For received activations, contains all information needed to receive and
    reconstruct a tensor, including the recv buffer and metadata.
    For root arguments, only stores metadata (no recv buffer needed).
    """

    def __init__(
        self,
        input_name: str,
        source: int | None,
        buffer: torch.Tensor | None,
        tensor_meta: TensorMeta | None,
        *,
        is_root_arg: bool = False,
    ):
        # Name of this input
        self.input_name = input_name
        # Stage index of the source of this input (None for root args)
        self.source = source
        # Buffer to receive the input into (None for root args)
        self.buffer = buffer
        # Tensor metadata for validation and DTensor reconstruction
        self.tensor_meta = tensor_meta
        # Whether this is a root-level model input (no recv needed)
        self.is_root_arg = is_root_arg

    def __repr__(self):
        if self.is_root_arg:
            return f"_RecvInfo(input={self.input_name}, root_arg=True)"
        meta_type = type(self.tensor_meta).__name__ if self.tensor_meta else "None"
        buffer_shape = self.buffer.size() if self.buffer is not None else "None"
        return f"_RecvInfo(input={self.input_name}, source={self.source}, shape={buffer_shape}, meta={meta_type})"


class _PipelineStageBase(ABC):
    """
    Base class for pipeline stages.
    Defines or implements common methods used by the `_PipelineStage` used by
    the tracing frontend and `PipelineStage` used by manual frontend.
    """

    def __init__(
        self,
        submodule: torch.nn.Module,
        stage_index: int,
        num_stages: int,
        device: torch.device,
        group: dist.ProcessGroup | None = None,
        dw_builder: Callable[[], Callable[..., None]] | None = None,
    ):
        """
        Args:
            submodule (torch.nn.Module): The module to be executed in this stage.
            stage_index (int): The index of this stage.
            num_stages (int): The total number of stages in this pipeline.
            device (torch.device): The device to run this stage on.
            group (Optional[dist.ProcessGroup]): The process group to use for communication.
                If `None`, the default process group will be used.
                Default: `None`.
            dw_builder (Optional[Callable[[], Callable[..., None]]): If provided, dw_builder is a builder function
                that will build a new dw_runner function that will run parts of module backward that were intentionally
                skipped during the module's actual backward pass. The builder must be invoked by stage after stage runs
                model backwards, and stage should save the latest dw_runner to run during weight pas (W).
                If not provided, a dw_runner will be generated automatically by traversing the autograd graph.
                When used with schedules that only have F and B steps, the fresh dw_runner function will be called as
                part of I (input backwards). When used with F,I,W schedules, the dw_runner function implements 'W'.
        """
        super().__init__()
        if stage_index >= num_stages:
            raise ValueError(
                f"Stage index {stage_index} is out of range of {num_stages}"
            )

        self.submod = submodule
        self.stage_index = stage_index
        self.num_stages = num_stages
        self.device = device
        self.group = group

        self.dw_builder = dw_builder

        # backward state
        self.backward_state: dict[int, tuple[Any, ...]] = {}

        # store dw_runner per microbatch_id
        self.dw_runner: dict[int, Callable[..., None]] = {}

        # `group_rank` is rank in process group `group`.
        self.group_rank = dist.get_rank(self.group)
        self.group_size = dist.get_world_size(self.group)
        if self.group_size > self.num_stages:
            raise RuntimeError(
                f"Pipeline group size {self.group_size} cannot be larger than number of stages {self.num_stages}"
            )

        # Run time states
        # map microbatch ID to list of forward tensor args
        self.fwd_cache: dict[int, tuple[Any, list[torch.Tensor]]] = {}
        # map microbatch ID to list of backward grad tensor args
        self.bwd_cache: dict[int, tuple[torch.Tensor | None, ...]] = {}
        # Caching chunk outputs for final output merge or reduction
        self.output_chunks: list[Any] = []

        # Initialize has_backward to false; this will be set to true if loss
        # function is passed to pipeline schedule
        self.has_backward = False
        # Log prefix
        self.log_prefix = f"[Stage {self.stage_index}]"

        # Forward infra
        self.args_recv_info: dict[int, tuple[_RecvInfo, ...]] = {}
        self.act_send_info: dict[int, list] = {}

        # Backward infra will created lazily
        self.grad_recv_info: dict = {}
        self.grad_send_info: list | None = None

        # To be populated later by the Schedule
        self.chunks: int | None = None
        self.stage_index_to_group_rank: dict[int, int] = {
            i: i % self.group_size for i in range(self.num_stages)
        }

        # DTensor support: mesh provider function for looking up DeviceMesh
        # Will be set by subclass or during inference
        self.get_mesh: (
            Callable[[tuple[str, ...], Optional[_MeshLayout]], DeviceMesh] | None
        ) = None

        # DTensor support: mesh cache for looking up DeviceMesh by (dim_names, layout)
        self._mesh_cache = _MeshCache()

        # DTensor support: consolidated stage metadata container
        # Contains inputs, outputs, input_grads, output_grads metadata
        self._stage_meta = _StageMeta()

    @property
    def has_backward(self) -> bool:
        """
        Returns true if this stage has a backward pass.
        """
        return self._has_backward

    @has_backward.setter
    def has_backward(self, has_backward: bool):
        self._has_backward = has_backward

    @property
    def is_first(self):
        """
        Returns true if this stage is the first stage in the pipeline.
        """
        return self.stage_index == 0

    @property
    def is_last(self):
        """
        Returns true if this stage is the last stage in the pipeline.
        """
        return self.stage_index == self.num_stages - 1

    def _check_chunk_id(self, chunk_id: int):
        if self.chunks is None:
            raise RuntimeError(
                "Attempted to access chunk_id before chunks have been configured."
            )
        if chunk_id >= self.chunks:
            raise RuntimeError(
                f"Chunk id {chunk_id} is out of range [0, {self.chunks})"
            )

    def _create_grad_send_info(
        self,
        args_recv_info: tuple,
    ) -> list[int | None]:
        """
        Create a list of stage indices to send gradients to.
        """
        grad_send_info: list[int | None] = []

        def map_recv_to_send(a):
            # Note: we send gradients back to previous stage as long as in
            # forward it is a received input, regardless of whether it requires
            # grad. It is up to the previous stage to discard this gradient.
            if a.is_root_arg:
                # Root args don't have a source stage to send gradients to
                grad_send_info.append(None)
                return None
            else:
                grad_send_info.append(a.source)
                return a.source

        map_aggregate(args_recv_info, map_recv_to_send)

        logger.debug("%s Grad send info: %s", self.log_prefix, grad_send_info)
        return grad_send_info

    @abstractmethod
    def _prepare_forward_infra(
        self,
        num_microbatches: int,
        args: Optional[Union[tuple[Any, ...], _StageForwardMeta]],
        kwargs: Optional[dict[str, Any]] = None,
        has_backward: bool = False,
        loss_fn: Optional[Callable[..., torch.Tensor]] = None,
        target: Optional[torch.Tensor] = None,
    ) -> Optional[_StageForwardMeta]:
        raise NotImplementedError

    def _prepare_backward_infra(
        self,
        num_microbatches: int,
        loss_fn: Optional[Callable[..., torch.Tensor]] = None,
        target: Optional[torch.Tensor] = None,
        received_grad_meta: Optional[_StageBackwardMeta] = None,
    ) -> Optional[_StageBackwardMeta]:
        """
        Run backward metadata inference and prepare backward infrastructure.

        Default implementation for stages that don't support DTensor backward
        metadata inference (e.g., traced frontend). Subclasses like PipelineStage
        override this for full DTensor support.

        Args:
            num_microbatches: Number of microbatches
            loss_fn: Loss function for last stage
            target: Target tensor for last stage
            received_grad_meta: Gradient metadata received from next stage

        Returns:
            None (override in subclasses for DTensor support)
        """
        # Default: just prepare backward recv info without DTensor metadata inference
        self._setup_backward_recv_info(num_microbatches)
        return None

    def _setup_backward_recv_info(self, num_microbatches: int):
        # TODO: this is needed for backward_maybe_with_nosync
        self.chunks = num_microbatches

        # IMPORTANT: _create_grad_recv_info reads self._stage_meta.output_grads
        # to attach DTensor metadata to _RecvInfo objects. The clear below MUST
        # happen after all _create_grad_recv_info calls complete.
        for mb_index in range(num_microbatches):
            # `grad_recv_info` is a mirror of `act_send_info`
            self.grad_recv_info[mb_index] = self._create_grad_recv_info(
                self.act_send_info
            )

        # Clear transient output grad DTensor metadata — now attached to _RecvInfo.
        # Safe to clear: all _create_grad_recv_info calls above have completed.
        self._stage_meta.output_grads = None

    @abstractmethod
    def _create_grad_recv_info(
        self,
        act_send_info: dict,
    ) -> tuple[_RecvInfo, ...]:
        raise NotImplementedError

    def _get_recv_ops(
        self,
        recv_infos: tuple[_RecvInfo, ...],
    ) -> list[dist.P2POp]:
        """
        Helper function shared by `get_fwd_recv_ops` and `get_bwd_recv_ops`.
        Returns a list of ops that correspond to the recv infos.
        """
        ops: list[dist.P2POp] = []
        for info in recv_infos:
            if info.is_root_arg:
                # Root args don't need recv operations
                continue

            # At this point, source and buffer are guaranteed non-None
            assert info.source is not None and info.buffer is not None  # noqa: S101
            peer_rank = self.stage_index_to_group_rank[info.source]
            peer_global_rank = (
                peer_rank
                if self.group is None
                else dist.get_global_rank(self.group, peer_rank)
            )
            ops.append(
                dist.P2POp(dist.irecv, info.buffer, peer_global_rank, self.group)
            )

        return ops

    """[Note: V-schedule special case]

    V-Schedules have a special case where 2 stages with adjacent stage_id are on the same rank.

    ex: 2 ranks, 4 stages forms a simple V:
    rank0:  stage 0                   stage 3
    rank1:          stage 1  stage 2

    stage 0,1 and 2,3 communicate activations using send/recv as usual, but stage 1,2 do not need to
    use communication ops.  Instead, they should pass tensor data directly via function call.

    set_local_fwd_input and (get_local_bwd_output + set_local_bwd_input) facilitate this optimization, and
    should be called at the appropriate time during the pipeline schedule (after forward or backward execution).
    """

    def set_local_fwd_input(self, prev_stage_outputs: Any, mb_index: int) -> None:
        """
        Moves 'prev_stage_outputs' from another stage on the same rank into place as inputs for this stage. Avoids
        copying tensor data or using send/recv op.  Detaches original tensor and sets requires_grad so the
        tensor can serve as a leaf for autograd and gradients can be collected from it during backward.
        Handles DTensor activations for V-schedule local passing.
        """
        recv_infos: tuple[_RecvInfo, ...] = self.args_recv_info[mb_index]

        # See [Note: pipeline model output type]
        prev_stage_outputs = _normalize_model_output_as_tuple(prev_stage_outputs)

        for info, tensor in zip(recv_infos, prev_stage_outputs, strict=True):
            if not isinstance(tensor, torch.Tensor):
                raise AssertionError(
                    f"expected tensor values as outputs from prev stage, got {type(tensor)}"
                )
            if info.is_root_arg:
                raise AssertionError(
                    "set_local_fwd_input should only be called on non-first stage, which should always have non-root RecvInfo"
                )

            # Pass the activation tensor directly (same rank for local execution).
            # Detach to create a new autograd leaf for the fresh autograd graph.
            info.buffer = to_local_if_dtensor(tensor).detach().requires_grad_(True)

    def get_local_bwd_output(self, mb_index):
        """
        Returns the input grad tensors for this stage, which correspond to the stage inputs during forward.
        """
        if not self.has_backward:
            raise AssertionError(
                "can't steal_bwd_input if this stage doesn't have backward"
            )
        if self.is_first:
            raise AssertionError("can't get bwd output if this stage is first")

        self._check_chunk_id(mb_index)
        return self.bwd_cache.pop(mb_index)

    def set_local_bwd_input(
        self, next_stage_bwd_outputs: tuple[torch.Tensor | None, ...], mb_index: int
    ) -> None:
        """
        Moves 'grad input' tensors from the next stage to 'grad_output' on this stage, avoiding a copy or send/recv.
        Does not detach or set '_requires_grad'.
        Handles DTensor gradients for V-schedule local passing.
        """
        if not isinstance(next_stage_bwd_outputs, tuple):
            raise AssertionError(f"Expected tuple, got {type(next_stage_bwd_outputs)}")

        if not self.has_backward:
            raise AssertionError(
                "can't set bwd input if this stage doesn't have backward"
            )
        if self.is_last:
            raise AssertionError("can't set bwd input if this stage is last")
        recv_infos = self.grad_recv_info[mb_index]
        for info, tensor in zip(recv_infos, next_stage_bwd_outputs, strict=True):
            if tensor is None:
                continue
            if not isinstance(tensor, torch.Tensor):
                raise AssertionError(
                    f"expected tensor values as outputs from prev stage, got {type(tensor)}"
                )
            if info.is_root_arg:
                raise AssertionError(
                    "set_local_bwd_input should only be called with non-root RecvInfo"
                )

            # Extract local tensor for the buffer (handles DTensor or plain tensor)
            info.buffer = to_local_if_dtensor(tensor)

    def get_fwd_recv_ops(self, fwd_chunk_id: int) -> list[dist.P2POp]:
        """
        Returns a list of ops that are needed to receive the input arguments
        for this stage.
        """
        recv_infos: tuple[_RecvInfo, ...] = self.args_recv_info[fwd_chunk_id]

        return self._get_recv_ops(recv_infos)

    def get_bwd_recv_ops(self, bwd_chunk_id: int) -> list[dist.P2POp]:
        """
        Returns a list of ops that are needed to receive the gradients
        for this stage.
        """
        if not self.has_backward or self.is_last:
            return []

        recv_infos = self.grad_recv_info[bwd_chunk_id]
        return self._get_recv_ops(recv_infos)

    def get_fwd_send_ops(self, fwd_chunk_id: int) -> list[dist.P2POp]:
        """
        Get the activation send ops for current stage's forward.
        Handles DTensor outputs by extracting local tensors.
        """
        output_tuple, _ = self.fwd_cache[fwd_chunk_id]

        ops: list[dist.P2POp] = []

        for idx, out in enumerate(output_tuple):
            dst_stages = self.act_send_info[idx]
            for dst in dst_stages:
                if dst is None:
                    continue
                # Extract local tensor if DTensor
                send_tensor = to_local_if_dtensor(out)
                logger.debug(
                    "%s Sending tensor to Stage %s: %s",
                    self.log_prefix,
                    dst,
                    send_tensor.size(),
                )
                peer_rank = self.stage_index_to_group_rank[dst]
                peer_global_rank = (
                    peer_rank
                    if self.group is None
                    else dist.get_global_rank(self.group, peer_rank)
                )
                ops.append(
                    dist.P2POp(dist.isend, send_tensor, peer_global_rank, self.group)
                )

        return ops

    def get_bwd_send_ops(self, bwd_chunk_id: int) -> list[dist.P2POp]:
        """
        Get the gradient send ops for current stage's backward.
        Handles DTensor gradients by extracting local tensors.
        """
        if not self.has_backward:
            return []

        self._check_chunk_id(bwd_chunk_id)
        # Create bwd send infra lazily
        if self.grad_send_info is None:
            # Send info for input grads during backward:
            # List of destinations corresponding to input grads
            # Can be None if an input has no grad
            # `grad_send_info` is a mirror of `args_recv_info`
            self.grad_send_info = self._create_grad_send_info(self.args_recv_info[0])

        ops: list[dist.P2POp] = []
        grads_input = self.bwd_cache.pop(bwd_chunk_id)
        # Validate backward output (input gradients) for DTensor metadata
        self._validate_bwd_output(grads_input)
        for grad, grad_recv_stage in zip(grads_input, self.grad_send_info, strict=True):
            if isinstance(grad, torch.Tensor) and grad_recv_stage is not None:
                # Extract local tensor if DTensor
                send_tensor = to_local_if_dtensor(grad)
                logger.debug(
                    "%s Sending gradient to Stage %s: %s",
                    self.log_prefix,
                    grad_recv_stage,
                    send_tensor.size(),
                )
                peer_rank = self.stage_index_to_group_rank[grad_recv_stage]
                peer_global_rank = (
                    peer_rank
                    if self.group is None
                    else dist.get_global_rank(self.group, peer_rank)
                )
                ops.append(
                    dist.P2POp(dist.isend, send_tensor, peer_global_rank, self.group)
                )
            else:
                if grad is not None or grad_recv_stage is not None:
                    raise RuntimeError(
                        f"[{self.stage_index}] for chunk {bwd_chunk_id} has gradients {grad} "
                        f"and is expecting to send gradients to stage {grad_recv_stage}"
                    )
        return ops

    def clear_runtime_states(self) -> None:
        """
        Clear runtime states of the stage.
        """
        # map microbatch ID to list of forward tensor args
        self.fwd_cache.clear()
        # Caching chunk outputs for final output merge or reduction
        self.output_chunks.clear()

        # Clear grad of input buffers in between schedule steps. This is because
        # `torch.autograd.backward()` will accumulate gradients into leaf
        # tensors by default. For gradients to pass back to previous stages, we
        # don't want such accumulation.
        for recv_tuple in self.args_recv_info.values():  # iterate over all chunks
            for a in recv_tuple:  # iterate over all input args
                if not a.is_root_arg and a.buffer is not None:
                    # Set to None is the newer and recommended way to clear grads, compared to `zero_()`.
                    # See https://github.com/pytorch/pytorch/pull/92731
                    a.buffer.grad = None

    def _map_tensor_from_recv_info(
        self,
        recv_infos: tuple[_RecvInfo, ...],
    ):
        """
        Map tensors from recv infos to a list.
        """

        def get_recv_tensor(info):
            if info.is_root_arg:
                raise AssertionError("Cannot get recv tensor from root arg")
            return info.buffer

        return map_aggregate(cast(Argument, recv_infos), get_recv_tensor)

    def _retrieve_recv_activations(
        self,
        fwd_chunk_id: int,
    ):
        """
        Retrieve the activations received for the current stage during forward.
        Reconstructs DTensors if the inputs were DTensors.
        Also validates DTensor metadata against expected values.
        """
        recv_infos = self.args_recv_info[fwd_chunk_id]

        activations = []
        for i, info in enumerate(recv_infos):
            if not info.is_root_arg:
                # Non-root args have valid buffer and tensor_meta
                assert info.buffer is not None  # noqa: S101
                assert info.tensor_meta is not None  # noqa: S101
                local_tensor = info.buffer

                if isinstance(info.tensor_meta, _DTensorMeta):
                    # Reconstruct DTensor from local tensor + metadata
                    mesh = self._mesh_cache.get_or_create(
                        info.tensor_meta.mesh_cache_key, self.get_mesh
                    )
                    activation = DTensor.from_local(
                        local_tensor,
                        device_mesh=mesh,
                        placements=info.tensor_meta.placements,
                        shape=info.tensor_meta.global_shape,
                        stride=info.tensor_meta.global_stride,
                        run_check=False,
                    )
                    # DTensor.from_local creates a non-leaf tensor (via view_as),
                    # so we need retain_grad() for backward to populate .grad
                    if activation.requires_grad:
                        activation.retain_grad()

                    # Validate the reconstructed DTensor against expected metadata
                    validate_metadata(
                        f"Stage {self.stage_index} forward input {i} (reconstructed)",
                        info.tensor_meta,
                        activation,
                        raise_on_mismatch=True,
                    )
                else:
                    activation = local_tensor

                activations.append(activation)
            else:
                raise AssertionError(f"Expected _RecvInfo but got {type(info)}")

        return tuple(activations)

    def _retrieve_recv_grads(
        self,
        bwd_chunk_id: int,
    ):
        """
        Retrieve the gradients received for the current stage during backward.
        Reconstructs DTensors if the activations were DTensors.
        Also validates DTensor metadata against expected values.
        Handles None gradients gracefully (for inputs that don't require grad).
        """
        recv_infos = self.grad_recv_info[bwd_chunk_id]

        grads: list[torch.Tensor | DTensor | None] = []
        for i, info in enumerate(recv_infos):
            if not info.is_root_arg:
                # Non-root args should have tensor_meta populated
                assert info.tensor_meta is not None  # noqa: S101
                local_tensor = info.buffer

                # Gradients can be None for non-differentiable outputs
                if local_tensor is None:
                    grads.append(None)
                    continue

                if isinstance(info.tensor_meta, _DTensorMeta):
                    # Reconstruct DTensor gradient from local tensor + metadata
                    mesh = self._mesh_cache.get_or_create(
                        info.tensor_meta.mesh_cache_key, self.get_mesh
                    )
                    grad = DTensor.from_local(
                        local_tensor,
                        device_mesh=mesh,
                        placements=info.tensor_meta.placements,
                        shape=info.tensor_meta.global_shape,
                        stride=info.tensor_meta.global_stride,
                        run_check=False,
                    )

                    # Validate the reconstructed DTensor against expected metadata
                    validate_metadata(
                        f"Stage {self.stage_index} backward grad {i} (reconstructed)",
                        info.tensor_meta,
                        grad,
                        raise_on_mismatch=True,
                    )
                else:
                    grad = local_tensor

                grads.append(grad)
            else:
                raise AssertionError(f"Expected _RecvInfo but got {type(info)}")

        return tuple(grads)

    def forward_maybe_with_nosync(self, *args, **kwargs):
        # If submod is wrapped with DDP, we use the `no_sync` context manager to
        # avoid gradient all-reduce per microbatch
        if isinstance(self.submod, DistributedDataParallel):
            with self.submod.no_sync():  # type: ignore[operator]
                out_val = self.submod(*args, **kwargs)
        else:
            out_val = self.submod(*args, **kwargs)
        return out_val

    def scale_grads(self, grad_scale_factor: int) -> None:
        """Scale gradients model gradients by `grad_scale_factor`, which should be specified in coordination with the
        loss function used with pipelining.  For loss functions which perform 'mean' loss reduction, `grad_scale_factor`
        should be set to num_microbatches.  For loss functions that use `sum` reduction, `grad_scale_factor` should
        be set to 1.

        Should only be called once per pipeline schedule step, after all backwards passes have completed.
        """

        # PP scales only for its own contribution (microbatches), but relies on DP to scale further
        # for DP degree.
        if grad_scale_factor != 1:
            for p in self.submod.parameters():
                if p.grad is not None:
                    p.grad.div_(grad_scale_factor)

    def backward_maybe_with_nosync(
        self,
        backward_type,
        bwd_kwargs: dict,
        last_backward: bool = False,
    ) -> tuple[tuple[torch.Tensor | None, ...], list[dict[str, Any]] | None]:
        """
        Whether using PP with FSDP, DDP, or replicate there are some runtime differences between the last backward step and the
        other steps.  Namely, we need to accumulate gradients on previous steps and reduce them on the last step, but
        there are additional state-variables and performance considerations depending on the data parallelism used.
        This helper should adapt any pipeline parallel schedule to work with common/supported data parallel libraries.
        """

        def perform_backward(
            backward_type,
        ) -> Callable[
            [],
            tuple[tuple[torch.Tensor | None, ...], list[dict[str, Any]] | None],
        ]:
            if backward_type == "full":
                return lambda: (
                    stage_backward(
                        bwd_kwargs["stage_output"],
                        bwd_kwargs["output_grads"],
                        bwd_kwargs["input_values"],
                    ),
                    None,
                )
            elif backward_type == "input":
                return lambda: stage_backward_input(
                    bwd_kwargs["stage_output"],
                    bwd_kwargs["output_grads"],
                    bwd_kwargs["input_values"],
                    self.submod.parameters(),
                )
            elif backward_type == "weight":
                return lambda: (
                    stage_backward_weight(
                        self.submod.parameters(), bwd_kwargs["param_groups"]
                    ),
                    None,
                )
            else:
                raise RuntimeError(f"Unknown backward type: {backward_type}")

        # If submod is wrapped by DDP
        if isinstance(self.submod, DistributedDataParallel):
            if last_backward:
                # Last chunk, prepare for gradient reduction
                # HACK: reaching into DDP implementation details here. Is there a better way?
                self.submod.reducer.prepare_for_backward(  # type: ignore[union-attr, operator]
                    list(
                        torch.nn.parallel.distributed._find_tensors(  # type: ignore[attr-defined]
                            bwd_kwargs["stage_output"]
                        )
                    )
                )
                result = perform_backward(backward_type)()
            else:
                with self.submod.no_sync():  # type: ignore[operator]
                    result = perform_backward(backward_type)()

        # If submod is a FSDP or replicate module
        elif isinstance(self.submod, FSDPModule):
            self.submod.set_is_last_backward(False)
            self.submod.set_reshard_after_backward(False)
            self.submod.set_requires_gradient_sync(False)
            result = perform_backward(backward_type)()

        else:
            # Non-DP submodule, regular backward
            result = perform_backward(backward_type)()

        grads, param_groups = result
        return grads, param_groups

    def forward_one_chunk(
        self,
        fwd_chunk_id: int,
        args: tuple[Any, ...],
        kwargs: dict[str, Any] | None = None,
        save_forward_output: bool = True,
    ):
        """
        Perform forward pass on the stage with one microbatch.
        `args` and `kwargs` are the inputs from *external* to this stage.
        As of Sept 2024:
        - `args` applies to the first stage only, other stages receives args
          through activation transmission.
        - `kwargs` can be passed to all stages via respective `step` calls.
        """

        if self.is_first:
            # First stage doesn't need to receive anything
            composite_args = args
        else:
            # Receive activations for this chunk
            # Activations only come in args form
            composite_args = self._retrieve_recv_activations(fwd_chunk_id)

        composite_kwargs = kwargs or {}

        self._validate_fwd_inputs(composite_args)

        # Compute forward
        try:
            output = self.forward_maybe_with_nosync(*composite_args, **composite_kwargs)

        except Exception as e:
            exc_msg = f"""
            {self.log_prefix} failed to run forward:
            args: {map_debug_info(composite_args)}
            kwargs: {map_debug_info(composite_kwargs)}
            """
            raise RuntimeError(exc_msg) from e

        # See [Note: pipeline model output type]
        output_tuple = _normalize_model_output_as_tuple(output)

        # Prepare for final output merge or reduction
        # Output chunks is only used for the last stage since we only merge the output of the last stage
        if self.is_last and save_forward_output:
            self.output_chunks.append(output)
        # Save activations and inputs for backward
        flat_args = flatten_args(composite_args)
        flat_kwargs = flatten_args(composite_kwargs)
        flatten_input_tensors = flat_args + flat_kwargs
        self.fwd_cache[fwd_chunk_id] = (
            output_tuple,  # stage_output
            flatten_input_tensors,  # input_values
        )

        logger.debug(
            "%s Forwarded chunk %s, outputs: %s",
            self.log_prefix,
            fwd_chunk_id,
            map_debug_info(output),
        )
        self._validate_fwd_outputs(output_tuple)

        # We return the original user-provided output, not normalized to tuple.
        # See [Note: pipeline model output type]
        return output

    def backward_one_chunk(
        self,
        bwd_chunk_id: int,
        loss=None,
        full_backward: bool = True,
        last_backward=False,
    ):
        """
        Perform backward pass on the module.
        This should only be called once per microbatch.

        If full_backward is True (the default), the full backward pass including weight and input gradients will be run,
        and it is an error to call `backward_weight_one_chunk` for this bwd_chunk_id.

        If full_backward is False, it is optional that `dw_runner` was provided to the PipelineStage at __init__ time,
        and a subsequent call to `backward_weight_one_chunk` is required to invoke dw_runner and complete the backward.

        last_backward is controlled by the schedule and signals synchronization of gradients across DP groups
        after the last backward.
        """
        # skip backward computation if backward is not enabled
        if not self.has_backward:
            return

        self._check_chunk_id(bwd_chunk_id)

        (
            stage_output,
            input_values,
        ) = self.fwd_cache.pop(bwd_chunk_id)

        # Compute backward
        if self.is_last:
            # Last stage computes gradients from loss and has no gradients from
            # next stage
            bwd_kwargs = {
                "stage_output": loss,
                "output_grads": None,
                "input_values": input_values,
            }
        else:
            # Otherwise, receive gradients from next stage
            grads_output = self._retrieve_recv_grads(bwd_chunk_id)
            # Validate backward input (output gradients) for DTensor metadata
            self._validate_bwd_input(bwd_chunk_id, grads_output)
            # If an input to the pipeline requires gradient,
            # `torch.autograd.backward` will accumulate the gradient into the
            # `.grad` field of such input
            bwd_kwargs = {
                "stage_output": stage_output,
                "output_grads": grads_output,
                "input_values": input_values,
            }

        grads_input: tuple[torch.Tensor | None, ...] = ()

        # Custom backward function
        if self.dw_builder:
            # TODO: We may want to change our semantics so we are allowed to ignore
            # the 'dw_builder' and call full_backward directly when it is a full_backward op.
            grads_input, _ = self.backward_maybe_with_nosync(
                "full",
                bwd_kwargs,
                last_backward=last_backward,
            )
            if full_backward:
                self.dw_builder()()
            else:
                self.dw_runner[bwd_chunk_id] = self.dw_builder()
        else:
            if full_backward:
                grads_input, _ = self.backward_maybe_with_nosync(
                    "full", bwd_kwargs, last_backward=last_backward
                )
            else:
                param_groups: list[dict[str, Any]] | None = None
                # Skip the backward for the first stage since we will perform the weight update with
                # autograd.backward in backward_weight_one_chunk
                if not self.is_first:
                    if isinstance(bwd_kwargs["stage_output"], torch.Tensor):
                        bwd_kwargs["stage_output"] = (bwd_kwargs["stage_output"],)

                    # perform the partial backwards for the inputs with a custom backward function
                    # when the "stage_ouput" is a loss, then it is a tensor, otherwise it is a tuple of tensors
                    grads_input, param_groups = self.backward_maybe_with_nosync(
                        "input", bwd_kwargs, last_backward=last_backward
                    )

                # TODO: we dont need to save this, add to dw_runner?
                self.backward_state[bwd_chunk_id] = (
                    bwd_kwargs["input_values"],
                    param_groups,
                    bwd_kwargs["stage_output"],
                    bwd_kwargs["output_grads"],
                )
                # Save a placeholder for the dw_runner
                self.dw_runner[bwd_chunk_id] = lambda: None

        self.bwd_cache[bwd_chunk_id] = grads_input

        if self.is_last and not self.is_first:
            # Autograd dependencies:
            #    rest_of_autograd_graph -> stage_output -> loss
            # stage_output is no longer used in the last stage for backward and only needed
            # to return to the user in merge_output_chunks, therefore
            # this should be detached to release autograd graph context and free memory earlier
            for t in stage_output:
                if not t._is_view():  # views are not detachable in-place
                    t.detach_()

        logger.debug("%s Backwarded chunk %s", self.log_prefix, bwd_chunk_id)

    def backward_weight_one_chunk(self, bwd_chunk_id: int, last_backward=False):
        # skip backward computation if backward is not enabled
        if not self.has_backward:
            return

        if bwd_chunk_id not in self.dw_runner:
            raise AssertionError(
                f"{self.log_prefix} Attempted to run backward_weight_one_chunk for chunk {bwd_chunk_id}"
                " without first calling `backward_one_chunk(full_backward=False)`"
            )

        if self.dw_builder is not None:
            self.dw_runner.pop(bwd_chunk_id)()
        else:
            (
                input_values,
                param_groups,
                stage_output,
                output_grads,
            ) = self.backward_state.pop(bwd_chunk_id)

            if self.stage_index != 0:
                bwd_kwargs = {
                    "stage_output": stage_output,
                    "param_groups": param_groups,
                }
                self.backward_maybe_with_nosync(
                    "weight", bwd_kwargs, last_backward=last_backward
                )
            else:
                # TODO: figure out a better way to do this:
                # if inputs does not require gradient,
                # then the parameter group will not be fully captured during stage_backward_input
                # in this case, we need call grad directly on the parameters
                # To solve: make input fn do the intersect compute and then finish it off during W
                bwd_kwargs = {
                    "stage_output": stage_output,
                    "output_grads": output_grads,
                    "input_values": input_values,
                }
                self.backward_maybe_with_nosync(
                    "full", bwd_kwargs, last_backward=last_backward
                )

    def _validate_fwd_inputs(self, args: tuple[torch.Tensor, ...]):
        """Raises a RuntimeError if this stage receives forward inputs of unexpected metadata."""
        inputs_meta = self._stage_meta.inputs
        if inputs_meta is None:
            return
        validate_tensors_metadata(
            f"Stage {self.stage_index} forward inputs",
            inputs_meta,
            args,
        )

    def _validate_fwd_outputs(self, outputs: tuple[torch.Tensor, ...]):
        """Raises a RuntimeError if this stage produces outputs of unexpected metadata."""
        outputs_meta = self._stage_meta.outputs
        if outputs_meta is None:
            return
        validate_tensors_metadata(
            f"Stage {self.stage_index} forward outputs",
            outputs_meta,
            outputs,
        )

    def _validate_bwd_input(
        self, bwd_chunk_id: int, output_grads: tuple[Optional[torch.Tensor], ...]
    ):
        """Validates DTensor metadata for output gradients received from the next stage."""
        # Get metadata from _stage_meta.output_grads
        output_grads_meta = self._stage_meta.output_grads
        if output_grads_meta is not None:
            # Use skip_none_actuals because grads can be None for non-differentiable outputs
            validate_tensors_metadata(
                f"Stage {self.stage_index} backward input (output_grads)",
                output_grads_meta,
                output_grads,
                skip_none_actuals=True,
            )
            return

        # Fallback: Access metadata from grad_recv_info
        grad_recv_infos = self.grad_recv_info.get(bwd_chunk_id)
        if grad_recv_infos is None:
            return

        for i, (recv_info, grad) in enumerate(
            zip(grad_recv_infos, output_grads, strict=True)
        ):
            if (
                not recv_info.is_root_arg
                and recv_info.tensor_meta is not None
                and grad is not None
            ):
                validate_metadata(
                    f"Stage {self.stage_index} backward input (output_grad) {i}",
                    recv_info.tensor_meta,
                    grad,
                    raise_on_mismatch=True,
                )

    def _validate_bwd_output(self, input_grads: tuple[Optional[torch.Tensor], ...]):
        """Validates DTensor metadata for input gradients being sent to the previous stage."""
        input_grads_meta = self._stage_meta.input_grads
        if input_grads_meta is None:
            return
        # Use skip_none_actuals because grads can be None for non-differentiable inputs
        validate_tensors_metadata(
            f"Stage {self.stage_index} backward output (input_grads)",
            input_grads_meta,
            input_grads,
            skip_none_actuals=True,
        )

    def _get_init_p2p_neighbors_ops(self) -> list[dist.P2POp]:
        """
        Get the operations to initialize the p2p communicators between previous and next stages.
        This is done so by creating a dummy tensor and sending it to the next stage and receiving
        from the previous stage.
        """
        ops: list[dist.P2POp] = []
        next_stage_peer_rank = self.stage_index_to_group_rank.get(self.stage_index + 1)
        prev_stage_peer_rank = self.stage_index_to_group_rank.get(self.stage_index - 1)

        recv_tensor = torch.zeros(1, device=self.device, dtype=torch.float32)
        send_tensor = torch.tensor(
            self.stage_index, device=self.device, dtype=torch.float32
        )
        # forward
        if not self.is_first:
            ops.append(
                dist.P2POp(
                    dist.irecv,
                    recv_tensor,
                    group_peer=prev_stage_peer_rank,
                    group=self.group,
                )
            )
        if not self.is_last:
            ops.append(
                dist.P2POp(
                    dist.isend,
                    send_tensor,
                    group_peer=next_stage_peer_rank,
                    group=self.group,
                )
            )

        # backward
        if not self.is_first:
            ops.append(
                dist.P2POp(
                    dist.isend,
                    send_tensor,
                    group_peer=prev_stage_peer_rank,
                    group=self.group,
                )
            )
        if not self.is_last:
            ops.append(
                dist.P2POp(
                    dist.irecv,
                    recv_tensor,
                    group_peer=next_stage_peer_rank,
                    group=self.group,
                )
            )

        return ops

    def perform_reduce_grad(self, grad_scale_factor: int):
        """
        Called as a part of schedule IR.
        REDUCE_GRAD action is scheduled after all microbatches W, B actions.

        Currently contains "post_backward" functionality for FSDP.
        We can try to extract post_backward in a separate IR action in future.
        """
        # Manually call post backward for FSDP
        if isinstance(self.submod, FSDPModule):
            fsdp_module = self.submod
            fsdp_module.set_is_last_backward(True)
            fsdp_module.set_reshard_after_backward(True)
            fsdp_module.set_requires_gradient_sync(True)

            if isinstance(fsdp_module, ReplicateModule):
                distributed_state = replicate.state(fsdp_module)  # type: ignore[arg-type]
            else:
                distributed_state = fully_shard.state(fsdp_module)  # type: ignore[attr-defined]

            for state in distributed_state._state_ctx.all_states:
                for fsdp_param_group in state._fsdp_param_groups:
                    fsdp_param_group.post_backward()

            # it would be much better if pipelining backward invoked .backward so autograd hooks
            # worked and modules like DDP/FSDP behaved as expected.  Working around this for the time being,
            # we need to call this too to ensure FSDP syncs its grad reduction ops back to the default stream.
            distributed_state._root_post_backward_final_callback()
        # Call gradient scaling at the end of the backward pass
        # NOTE: this must happen after FSDP post_backward is FSDP is enabled
        if grad_scale_factor != 1:
            self.scale_grads(grad_scale_factor)


class _PipelineStage(_PipelineStageBase):
    def __init__(
        self,
        stage_module: torch.nn.Module,
        stage_index: int,
        pipe_info: PipeInfo,
        device: torch.device,
        group: dist.ProcessGroup | None = None,
    ):
        """
        Create a pipeline stage given a stage_module to be wrapped by this stage
        and a `pipe_info` describing the stage relationship of the pipeline.

        Args:
            stage_module (torch.nn.Module): the module to be wrapped by this stage
            stage_index (int): the index of this stage in the pipeline
            pipe_info (PipeInfo): information about the pipeline, can be retrieved by `pipe.info()`
            device (torch.device): the device to be used by this stage
            group (Optional[dist.ProcessGroup]): the process group to be used by this stage
        """
        _PipelineStageBase.__init__(
            self,
            stage_module,
            stage_index,
            pipe_info.num_stages,
            device,
            group,
        )
        self.pipe_info = pipe_info

        # Find stage nodes in graph
        submod_nodes = [
            node for node in pipe_info.graph.nodes if node.op == "call_module"
        ]
        if len(submod_nodes) != self.num_stages:
            raise AssertionError(
                f"Number of submodules in pipe graph {len(submod_nodes)} does not match number of stages {self.num_stages}"
            )

        # Find my stage node in graph
        self.node = submod_nodes[self.stage_index]
        self.name = self.node.name
        logger.info(
            "[%s] Creating PipelineStage %s for %s",
            self.group_rank,
            stage_index,
            self.name,
        )

        # Create mapping from stage name to stage index
        self.submod_to_stage_index: dict[str, int] = {}
        for i, node in enumerate(submod_nodes):
            self.submod_to_stage_index.setdefault(node.name, i)

        # Cast submodule to device
        self._move_submod_to_device()

    def _move_submod_to_device(self):
        # Move submodule to indicated device if possible
        # Note: we cannot move meta module to real devices because meta tensors
        # do not support to() method. One needs to do an in-place tensor swap in
        # that case.
        has_meta_param = any(
            isinstance(p, FakeTensor) or p.is_meta for p in self.submod.parameters()
        )
        if has_meta_param:
            logger.debug("%s Found meta parameters!", self.log_prefix)
        else:
            self.submod.to(self.device)

    def _prepare_forward_infra(
        self,
        num_microbatches: int,
        args: Optional[Union[tuple[Any, ...], _StageForwardMeta]],
        kwargs: Optional[dict[str, Any]] = None,
        has_backward: bool = False,
        loss_fn: Optional[Callable[..., torch.Tensor]] = None,
        target: Optional[torch.Tensor] = None,
    ) -> Optional[_StageForwardMeta]:
        # TODO(whc)
        # this method should be deleted once lazy buffer allocation is implemented
        # for now, it ignores args/kwargs because it should not need to do shape inference
        for chunk in range(num_microbatches):
            self.args_recv_info[chunk] = self._create_act_recv_info()

        # Send info during forward for each activation
        self.act_send_info = self._create_act_send_info()
        return None

    def get_stage_index_of_submod(
        self,
        submod_name: str,
    ):
        """
        Given a submodule name, return the stage index of the submodule.
        """
        if submod_name not in self.submod_to_stage_index:
            raise AssertionError(f"Stage id of {submod_name} not found")

        return self.submod_to_stage_index[submod_name]

    def _create_act_recv_info(
        self,
    ):
        """
        Create a tuple of `_RecvInfo` for inputs to the stage.
        Uses metadata from _stage_meta.inputs for tensor metadata.
        """
        # Get input metadata from _stage_meta
        inputs_meta = self._stage_meta.inputs

        def create_recv_tensor(placeholder, arg_node, input_idx: int):
            """
            Create a receive buffer for a placeholder.
            """
            example_value = placeholder.meta["val"]
            if arg_node.op == "placeholder":
                # This is a root level placeholder, thus an input argument to the entire model.
                # We are likely at stage 0, hence no need to create a receive buffer.
                # Get tensor metadata from _stage_meta.inputs if available
                tensor_meta = None
                if inputs_meta is not None and input_idx < len(inputs_meta):
                    tensor_meta = inputs_meta[input_idx]
                return _RecvInfo(
                    input_name=f"root_input_{input_idx}",
                    source=None,
                    buffer=None,
                    tensor_meta=tensor_meta,
                    is_root_arg=True,
                )

            # Figure out the source stage of this input
            while arg_node.target is operator.getitem:
                # If the input is a getitem, we need to go deeper
                arg_node = arg_node.args[0]

            if arg_node.op != "call_module":
                raise AssertionError(f"Expecting call_module, got {arg_node.op}")
            src_stage = self.get_stage_index_of_submod(arg_node.name)

            # Get tensor metadata from _stage_meta.inputs
            if inputs_meta is None or input_idx >= len(inputs_meta):
                raise PipeliningMetadataError(
                    f"Stage {self.stage_index}: Missing input metadata for input {input_idx}. "
                    f"Ensure _stage_meta.inputs is populated before creating recv info."
                )
            tensor_meta = inputs_meta[input_idx]
            if tensor_meta is None:
                raise PipeliningMetadataError(
                    f"Stage {self.stage_index}: Input metadata at index {input_idx} is None. "
                    f"All inputs must have valid metadata."
                )

            # Create a receive buffer for this placeholder
            logger.debug(
                "%s Creating recv buffer for input '%s' : %s, %s",
                self.log_prefix,
                placeholder.name,
                example_value.shape,
                example_value.dtype,
            )
            buffer = _make_tensor_from_meta(example_value, self.device)
            # In case there is backward pass, set requires_grad for receive buffers
            # before first forward
            if self.has_backward:
                buffer.requires_grad_(True)

            return _RecvInfo(
                arg_node.name,
                src_stage,
                buffer,
                tensor_meta,
            )

        args_recv_info: list[_RecvInfo] = []
        # Filter out placeholder nodes from `self.submod` (a GraphModule)
        placeholders = filter(  # type: ignore[var-annotated]
            lambda node: node.op == "placeholder",  # type: ignore[arg-type]
            self.submod.graph.nodes,  # type: ignore[arg-type,union-attr]
        )
        # `placeholders` are nodes internal to submod.
        # `self.node.args` are dependency nodes in the outer graph.
        # The two are 1:1.
        for input_idx, (placeholder, arg_node) in enumerate(
            zip(placeholders, self.node.args, strict=True)
        ):
            # Create a receive buffer for this placeholder
            recv_info = create_recv_tensor(placeholder, arg_node, input_idx)
            args_recv_info.append(recv_info)

        logger.debug(
            "%s Activation recv / args info: %s", self.log_prefix, args_recv_info
        )
        # `args` is a Tuple, hence we will return a Tuple[_RecvInfo]
        return tuple(args_recv_info)

    def find_dst_rank(
        self,
        user: fx.Node,
    ) -> int | None:
        """
        Find the destination rank of a `user` node.
        If the `user` is not a submod, `None` may be returned.
        """
        if user.op == "call_module":
            # User is a stage (`call_module`)
            return self.get_stage_index_of_submod(user.name)
        else:
            # - If user.op == "output":
            #   No need to send back to rank 0
            # - If user.target is stage_backward:
            #   No need to send assuming submod output is stored locally or
            #   should be re-calculated in case of activation checkpointing
            return None

    def _create_act_send_info(self):
        """
        Create a dict of send info for activations.
        The dict is of the form:
        {
            output_index: [dst_rank_0, dst_rank_1, ...],
            ...
        }
        where the list of `dst_rank`s covers the case where an output value may
        be consumed by multiple stages.
        """
        # Output index: List of receiver ranks
        act_send_info: dict[int, list] = {}
        out_idx = 0

        for user in self.node.users:
            if user.target is operator.getitem:
                # Recursively find the real destination
                gi_dsts = act_send_info.setdefault(out_idx, [])
                for gi_user in user.users:
                    dst_rank = self.find_dst_rank(gi_user)
                    if dst_rank is not None:
                        gi_dsts.append(dst_rank)
                # Next `getitem` will point to the next output index
                out_idx += 1
            else:
                # In case of single output value, `out_idx` will not increase
                dsts = act_send_info.setdefault(out_idx, [])
                dst_rank = self.find_dst_rank(user)
                if dst_rank is not None:
                    dsts.append(dst_rank)

        output_node = self._get_output_node()
        output_vals: tuple[torch.Tensor] = tuple(
            v.meta["val"] for v in flatten_args(output_node.args)
        )
        # Extract and store output metadata
        self._stage_meta.outputs = extract_tensor_metas(output_vals)

        logger.debug("%s Send info: %s", self.log_prefix, act_send_info)
        return act_send_info

    def _get_output_node(self):
        output_nodes = [node for node in self.submod.graph.nodes if node.op == "output"]  # type: ignore[union-attr]
        if len(output_nodes) != 1:
            raise AssertionError(f"Expected 1 output node, got {len(output_nodes)}")
        output_node = output_nodes[0]
        return output_node

    def _create_grad_recv_info(
        self,
        act_send_info: dict,
    ) -> tuple[_RecvInfo, ...]:
        """
        Create a tuple of `_RecvInfo` for gradients.

        Note: DTensor support is NOT available for the traced pipeline frontend (_PipelineStage).
        DTensor metadata inference requires runtime information that is not available in the
        traced graph's example values. Use the manual PipelineStage frontend for DTensor support.
        """
        # Dict[output_index, _RecvInfo]
        grad_recv_info: dict[int, _RecvInfo] = {}
        output_node = self._get_output_node()

        # The output node may take multiple args, meaning the submod having multiple output values.
        output_vals = flatten_args(output_node.args)

        for out_idx, dst_list in act_send_info.items():
            if not dst_list:
                # No actual receiver for activation so no grad coming back
                continue

            output = output_vals[out_idx]
            example_value = output.meta["val"]

            # DTensors are not supported in the traced frontend — gradient
            # DTensor metadata cannot be inferred from the traced graph's
            # example values, so reconstructed gradients would silently lose
            # their placement information.
            if isinstance(example_value, DTensor):
                raise PipeliningMetadataError(
                    f"{self.log_prefix} DTensor detected in traced pipeline output '{output.name}'. "
                    f"DTensor metadata propagation is NOT supported for the traced frontend "
                    f"(_PipelineStage). Use the manual PipelineStage frontend for full DTensor support."
                )

            logger.debug(
                f"{self.log_prefix} Creating grad recv buffer for output {output.name} "  # noqa: G004
                f": {example_value.shape}, {example_value.dtype}"
            )

            # TODO: otherwise needs grad accumulation
            if len(dst_list) != 1:
                raise AssertionError("Backward of skip connections not supported yet")
            grad_src = dst_list[0]

            # Create tensor metadata from the example value
            # For traced frontend, this is always a plain tensor (DTensors are rejected above)
            tensor_meta = _TensorMeta.from_tensor(example_value)

            grad_recv_info[out_idx] = _RecvInfo(
                f"{grad_src}",  # noqa: G004
                grad_src,
                _make_tensor_from_meta(example_value, self.device),
                tensor_meta,
            )

        # Convert to tuple for convenience in get_ops and retrieve tensor
        grad_recv_info_tuple = tuple(grad_recv_info.values())
        logger.debug("%s Grad recv info: %s", self.log_prefix, grad_recv_info_tuple)
        return grad_recv_info_tuple


# A helper function to create a pipeline stage based on traced pipeline information
def build_stage(
    stage_module: torch.nn.Module,
    stage_index: int,
    pipe_info: PipeInfo,
    device: torch.device,
    group: dist.ProcessGroup | None = None,
) -> _PipelineStage:
    """
    Create a pipeline stage given a stage_module to be wrapped by this stage
    and pipeline information.

    Args:
        stage_module (torch.nn.Module): the module to be wrapped by this stage
        stage_index (int): the index of this stage in the pipeline
        pipe_info (PipeInfo): information about the pipeline, can be retrieved by `pipe.info()`
        device (torch.device): the device to be used by this stage
        group (Optional[dist.ProcessGroup]): the process group to be used by this stage

    Returns:
        _PipelineStage: a pipeline stage that can run with `PipelineSchedules`.
    """
    return _PipelineStage(
        stage_module,
        stage_index,
        pipe_info,
        device,
        group,
    )


class PipelineStage(_PipelineStageBase):
    """
    A class representing a pipeline stage in a pipeline parallelism setup.

    PipelineStage assumes sequential partitioning of the model, i.e. the model is split into chunks where outputs from
    one chunk feed into inputs of the next chunk, with no skip connections.

    PipelineStage supports both static and dynamic metadata inference:

    **Static Metadata Inference:**

    1. **Plain Tensor Mode** (input_args and output_args provided, no DTensors):
       - Gradient shapes are inferred from tensor shapes
       - input_grads/output_grads are ignored if provided

    2. **DTensor Mode** (input_args and output_args provided with DTensors, plus input_grads and output_grads):
       - All four arguments must be provided as tuples (use tuple of Nones for grads if not required)
       - For each DTensor input/output with requires_grad=True, corresponding grad must be DTensor
       - Mesh is extracted directly from provided DTensors

    **Dynamic Metadata Inference:**

    - If neither input_args nor output_args provided: Full runtime inference
    - If DTensors present but grads not fully specified: DYNAMIC mode (forward metadata may be static, backward inferred at runtime)

    Args:
        submodule (nn.Module): The PyTorch module wrapped by this stage.
        stage_index (int): The ID of this stage.
        num_stages (int): The total number of stages.
        device (torch.device): The device where this stage is located.
        input_args (Union[torch.Tensor, DTensor, Tuple], optional): The input arguments for the submodule.
            When DTensors are provided, their mesh is extracted and cached in metadata.
        output_args (Union[torch.Tensor, DTensor, Tuple], optional): The output arguments for the submodule.
            When DTensors are provided, their mesh is extracted and cached in metadata.
        output_grads (Union[torch.Tensor, DTensor, Tuple], optional): The gradient of outputs (received from next stage).
            Required for static backward metadata when using DTensors. Must be tuple (use Nones for non-grad tensors).
        input_grads (Union[torch.Tensor, DTensor, Tuple], optional): The gradient of inputs (sent to previous stage).
            Required for static backward metadata when using DTensors. Must be tuple (use Nones for non-grad tensors).
        group (dist.ProcessGroup, optional): The process group for distributed training. If None, default group.
        dw_builder (Optional[Callable[[], Callable[..., None]]): If provided, dw_builder will build a new dw_runner function
            that will the W action (input weights) for F, I, W (Fwd, Input, Weight) zero bubble schedules.
        get_mesh (Callable[[tuple[str, ...], tuple[int, ...] | None], DeviceMesh], optional): A function that returns
            a DeviceMesh given mesh dimension names and optional layout. Required for dynamic/deferred DTensor inference.
            The function signature is: (dim_names: tuple[str, ...], layout: _MeshLayout | None) -> DeviceMesh.
            Ignored if static DTensor metadata is fully specified (mesh extracted from provided DTensors).
    """

    # Type alias for mesh provider function
    MeshProvider = Callable[[tuple[str, ...], _MeshLayout | None], DeviceMesh]

    def __init__(
        self,
        submodule: nn.Module,
        stage_index: int,
        num_stages: int,
        device: torch.device,
        input_args: torch.Tensor | tuple[torch.Tensor, ...] | None = None,
        output_args: torch.Tensor | tuple[torch.Tensor, ...] | None = None,
        output_grads: torch.Tensor | tuple[torch.Tensor, ...] | None = None,
        input_grads: torch.Tensor | tuple[torch.Tensor, ...] | None = None,
        group: dist.ProcessGroup | None = None,
        dw_builder: Callable[[], Callable[..., None]] | None = None,
        get_mesh: MeshProvider | None = None,
    ):
        super().__init__(submodule, stage_index, num_stages, device, group, dw_builder)

        self.inputs_meta: tuple[torch.Tensor, ...] | None = None
        self.outputs_grad: list[torch.Tensor] = []
        self.get_mesh = get_mesh
        self._user_meta: _StageMeta | None = None
        self._inference_mode: InferenceMode | None = None
        self._fwd_outputs_for_bwd_meta: tuple[torch.Tensor, ...] | None = None
        self._fwd_inputs_for_bwd_meta: tuple[torch.Tensor, ...] | None = None
        self._fwd_kwargs_tensors_for_bwd_meta: tuple[torch.Tensor, ...] | None = None
        self._num_positional_inputs: int | None = None

        # Normalize args to tuples
        inputs = normalize_to_tuple(input_args)
        outputs = normalize_to_tuple(output_args)
        in_grads = normalize_to_tuple(input_grads)
        out_grads = normalize_to_tuple(output_grads)

        self._init_meta = _StageMeta(
            inputs=extract_tensor_metas(inputs) if inputs else None,
            outputs=extract_tensor_metas(outputs) if outputs else None,
            input_grads=extract_tensor_metas(in_grads) if in_grads else None,
            output_grads=extract_tensor_metas(out_grads) if out_grads else None,
        )

        # Cache meshes from user-provided DTensors
        for args in (inputs, outputs, in_grads, out_grads):
            if args is not None:
                self._mesh_cache.update_from_tensors(args)

        # Validate DTensor↔grad correspondence independently for inputs and outputs
        if self._init_meta.has_dtensors():
            if inputs and in_grads:
                validate_static_dtensor_grad_correspondence(
                    self.stage_index, inputs, in_grads, is_input=True
                )
            if outputs and out_grads:
                validate_static_dtensor_grad_correspondence(
                    self.stage_index, outputs, out_grads, is_input=False
                )

    def _recv_meta(self, src_stage: int) -> Any:
        """Receive metadata object from a stage on a different rank via P2P."""
        objects: list[Any] = [None]
        dist.recv_object_list(
            objects,
            src=dist.get_global_rank(
                self.group or dist.distributed_c10d._get_default_group(),
                self.stage_index_to_group_rank[src_stage],
            ),
            group=self.group,
            device=self.device,
            use_batch=True,
        )
        return objects[0]

    def _send_meta(self, meta: Any, dst_stage: int) -> None:
        """Send metadata object to a stage on a different rank via P2P."""
        dist.send_object_list(
            [meta],
            dst=dist.get_global_rank(
                self.group or dist.distributed_c10d._get_default_group(),
                self.stage_index_to_group_rank[dst_stage],
            ),
            group=self.group,
            device=self.device,
            use_batch=True,
        )

    def _is_same_rank(self, other_stage: int) -> bool:
        """Check if another stage is on the same rank as this stage."""
        return self.stage_index_to_group_rank[other_stage] == self.group_rank

    def _compute_input_grads_meta(
        self,
        outputs: tuple[torch.Tensor, ...],
        fwd_inputs: tuple[torch.Tensor, ...],
        grad_outputs: list[torch.Tensor] | None = None,
    ) -> tuple[TensorMeta | None, ...]:
        """
        Compute input gradient metadata using autograd.grad.

        Returns a tuple of TensorMeta with None for inputs that don't require grad.
        """
        # Get tensor inputs that require gradients, tracking their original positions
        grad_input_indices: list[int] = []
        tensor_inputs: list[torch.Tensor] = []
        for i, inp in enumerate(fwd_inputs):
            if isinstance(inp, (torch.Tensor, DTensor)) and inp.requires_grad:
                grad_input_indices.append(i)
                tensor_inputs.append(inp)

        if not tensor_inputs:
            return tuple()

        input_grads = torch.autograd.grad(
            outputs=outputs,
            inputs=tensor_inputs,
            grad_outputs=grad_outputs,
            allow_unused=True,
        )

        # Expand to full input length, placing None for inputs that don't require grad
        full_input_grads: list[TensorMeta | None] = [None] * len(fwd_inputs)
        for idx, g in zip(grad_input_indices, input_grads):
            full_input_grads[idx] = (
                extract_tensor_meta(g) if isinstance(g, torch.Tensor) else None
            )
        return tuple(full_input_grads)

    def _to_meta_input(self, arg: torch.Tensor) -> torch.Tensor:
        """
        Convert a tensor to meta device, preserving DTensor semantics.

        For DTensors: Creates a meta DTensor (DTensor with meta local tensor).
        For plain tensors: Creates a plain meta tensor.

        This is essential for forward metadata inference - if we convert DTensors
        to plain tensors, models with DTensor operations will produce wrong outputs.
        """
        if isinstance(arg, DTensor):
            local_meta = arg.to_local().to("meta")
            return DTensor.from_local(
                local_meta,
                device_mesh=arg.device_mesh,
                placements=arg.placements,
                shape=arg.shape,
                stride=arg.stride(),
                run_check=False,
            )
        else:
            return arg.to("meta")

    def _meta_from_metadata(self, meta: TensorMeta | None) -> torch.Tensor | None:
        """
        Reconstruct a meta tensor/DTensor from metadata.

        For _DTensorMeta: Creates a meta DTensor with proper placements.
        For _TensorMeta: Creates a plain meta tensor.
        """
        if meta is None:
            return None
        if isinstance(meta, _DTensorMeta):
            mesh = self._mesh_cache.get_or_create(meta.mesh_cache_key, self.get_mesh)
            return meta.to_meta_dtensor(mesh)
        else:
            return meta.to_meta_tensor()

    def _ones_from_metadata(self, meta: TensorMeta) -> torch.Tensor:
        """
        Create a ones tensor from metadata on meta device.

        For _DTensorMeta: Creates a meta DTensor filled with ones.
        For _TensorMeta: Creates a plain meta tensor filled with ones.

        Used for constructing grad_outputs in backward metadata inference.
        """
        local_ones = torch.ones(
            meta.shape,
            dtype=meta.dtype,
            device="meta",
            requires_grad=meta.requires_grad,
        )
        if isinstance(meta, _DTensorMeta):
            mesh = self._mesh_cache.get_or_create(meta.mesh_cache_key, self.get_mesh)
            return DTensor.from_local(
                local_ones,
                device_mesh=mesh,
                placements=meta.placements,
                shape=meta.global_shape,
                stride=meta.global_stride,
                run_check=False,
            )
        return local_ones

    def _forward_metadata_inference(
        self,
        args: tuple[Any, ...] | _StageForwardMeta | None,
        kwargs: dict[str, Any] | None = None,
        has_backward: bool = False,
    ) -> _StageForwardMeta | None:
        """
        Forward metadata inference: Stage 0 → N.

        Contract: _StageForwardMeta flows between stages.
        - First stage: receives real tensors, extracts metadata
        - Other stages: receive _StageForwardMeta (same-rank via arg, cross-rank via P2P)
        - All stages: return _StageForwardMeta for next stage (or send via P2P)

        DTensor handling: DTensors are preserved as DTensors (with meta local tensors)
        so that models with DTensor operations produce correct DTensor outputs.
        """
        kwargs = kwargs or {}

        # === RECEIVE: Get input metadata and create meta tensors ===
        if self.is_first:
            # First stage: extract metadata from real tensors
            if args is None or isinstance(args, _StageForwardMeta):
                raise RuntimeError(
                    f"Stage {self.stage_index}: First stage requires real tensors, "
                    f"got {type(args).__name__}."
                )
            tensor_args = tuple(
                a for a in args if isinstance(a, (torch.Tensor, DTensor))
            )
            self._stage_meta.inputs = extract_tensor_metas(tensor_args)
            # Convert to meta, preserving DTensor semantics
            meta_args = tuple(self._to_meta_input(a) for a in tensor_args)
        elif self._is_same_rank(self.stage_index - 1):
            # Same-rank: _StageForwardMeta passed via argument
            if not isinstance(args, _StageForwardMeta):
                raise RuntimeError(
                    f"Stage {self.stage_index}: Expected _StageForwardMeta from same-rank "
                    f"previous stage, got {type(args).__name__}."
                )
            self._stage_meta.inputs = args.forward_metas
            meta_args = tuple(self._meta_from_metadata(m) for m in args.forward_metas)
        else:
            # Cross-rank: receive _StageForwardMeta via P2P
            recv_meta = self._recv_meta(self.stage_index - 1)
            if not isinstance(recv_meta, _StageForwardMeta):
                raise RuntimeError(
                    f"Stage {self.stage_index}: Expected _StageForwardMeta from P2P, "
                    f"got {type(recv_meta).__name__}."
                )
            self._stage_meta.inputs = recv_meta.forward_metas
            meta_args = tuple(
                self._meta_from_metadata(m) for m in recv_meta.forward_metas
            )

        # Cache input shapes for recv buffer allocation
        self.inputs_meta = meta_args  # type: ignore[assignment]

        # === COMPUTE: Run forward pass on meta tensors/DTensors ===
        fwd_inputs = [x for x in meta_args if isinstance(x, (torch.Tensor, DTensor))]

        # Convert kwargs tensors to meta device (preserving DTensor semantics)
        meta_kwargs = {
            k: self._to_meta_input(v) if isinstance(v, (DTensor, torch.Tensor)) else v
            for k, v in kwargs.items()
        }

        # Run forward
        grad_context = torch.enable_grad() if has_backward else torch.no_grad()
        with grad_context:
            outputs = self.submod(*fwd_inputs, **meta_kwargs)

        # Normalize outputs to tuple
        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)
        elif isinstance(outputs, list):
            outputs = tuple(outputs)

        self._stage_meta.outputs = extract_tensor_metas(outputs)

        # Store for backward metadata inference
        # Note: We track kwargs tensors separately for gradient computation,
        # but only positional input grads flow to the previous stage
        fwd_kwargs_tensors = tuple(
            v for v in flatten_args(meta_kwargs) if isinstance(v, torch.Tensor)
        )
        self._fwd_outputs_for_bwd_meta = outputs
        self._fwd_inputs_for_bwd_meta = tuple(fwd_inputs)
        self._fwd_kwargs_tensors_for_bwd_meta = fwd_kwargs_tensors
        self._num_positional_inputs = len(fwd_inputs)

        # === SEND: Pass output metadata to next stage ===
        fwd_meta = _StageForwardMeta(forward_metas=self._stage_meta.outputs)

        if self.is_last or self._is_same_rank(self.stage_index + 1):
            # Same-rank or last: return for caller to pass
            return fwd_meta
        else:
            # Cross-rank: send via P2P
            self._send_meta(fwd_meta, self.stage_index + 1)
            return None

    def _backward_metadata_inference(
        self,
        loss_fn: Callable[..., torch.Tensor] | None = None,
        target: torch.Tensor | None = None,
        received_grad_meta: _StageBackwardMeta | None = None,
    ) -> _StageBackwardMeta | None:
        """
        Backward metadata inference: Stage N → 0.

        Contract: _StageBackwardMeta flows between stages.
        - Last stage: computes grads from loss, creates metadata
        - Other stages: receive _StageBackwardMeta (same-rank via arg, cross-rank via P2P)
        - All stages: return _StageBackwardMeta for prev stage (or send via P2P)
        """
        outputs = self._fwd_outputs_for_bwd_meta
        fwd_inputs = self._fwd_inputs_for_bwd_meta
        if outputs is None or fwd_inputs is None:
            raise RuntimeError(
                "Backward metadata inference requires forward metadata inference to run first"
            )

        # === RECEIVE: Get output grad metadata (except last stage) ===
        if self.is_last:
            if loss_fn is None or target is None:
                raise RuntimeError(
                    f"Stage {self.stage_index}: loss_fn and target required for last stage"
                )
            # Last stage: compute loss and input grads directly
            loss = loss_fn(outputs[0] if len(outputs) == 1 else outputs, target)
            self._stage_meta.output_grads = None
            self._stage_meta.input_grads = self._compute_input_grads_meta(
                (loss,), fwd_inputs
            )
        else:
            # Non-last stage: receive grad metadata from next stage
            if self._is_same_rank(self.stage_index + 1):
                # Same-rank: _StageBackwardMeta passed via argument
                if not isinstance(received_grad_meta, _StageBackwardMeta):
                    raise RuntimeError(
                        f"Stage {self.stage_index}: Expected _StageBackwardMeta from same-rank "
                        f"next stage, got {type(received_grad_meta).__name__}."
                    )
                self._stage_meta.output_grads = received_grad_meta.backward_metas
            else:
                # Cross-rank: receive _StageBackwardMeta via P2P
                recv_meta = self._recv_meta(self.stage_index + 1)
                if not isinstance(recv_meta, _StageBackwardMeta):
                    raise RuntimeError(
                        f"Stage {self.stage_index}: Expected _StageBackwardMeta from P2P, "
                        f"got {type(recv_meta).__name__}."
                    )
                self._stage_meta.output_grads = recv_meta.backward_metas

            # === COMPUTE: Build grad_outputs and compute input grads ===
            # Extract output tensors and corresponding grad_outputs from metadata
            # Must iterate together to maintain alignment
            if self._stage_meta.output_grads is None:
                raise PipeliningMetadataError(
                    f"Stage {self.stage_index}: output_grads metadata is required for backward inference."
                )
            stage_output_grads = self._stage_meta.output_grads

            output_tensors: list[torch.Tensor] = []
            grad_outputs_list: list[torch.Tensor] = []

            for idx, o in enumerate(outputs):
                if not isinstance(o, (torch.Tensor, DTensor)):
                    continue
                # Skip outputs without grad metadata (e.g., non-differentiable outputs)
                if idx >= len(stage_output_grads) or stage_output_grads[idx] is None:
                    continue
                output_tensors.append(o)
                grad_meta = stage_output_grads[idx]
                # grad_meta is guaranteed non-None due to the check above
                grad_outputs_list.append(self._ones_from_metadata(grad_meta))  # type: ignore[arg-type]

            if output_tensors:
                # Include kwargs tensors for gradient computation
                kwargs_tensors = self._fwd_kwargs_tensors_for_bwd_meta or ()
                all_fwd_inputs = fwd_inputs + kwargs_tensors
                all_input_grads = self._compute_input_grads_meta(
                    tuple(output_tensors), all_fwd_inputs, grad_outputs_list
                )
                # Only positional input grads flow to previous stage
                num_pos = self._num_positional_inputs or len(fwd_inputs)
                self._stage_meta.input_grads = all_input_grads[:num_pos]
            else:
                self._stage_meta.input_grads = tuple()

        # Clean up temporary storage
        self._fwd_outputs_for_bwd_meta = None
        self._fwd_inputs_for_bwd_meta = None
        self._fwd_kwargs_tensors_for_bwd_meta = None
        self._num_positional_inputs = None

        # === SEND: Pass input grad metadata to previous stage ===
        bwd_meta = _StageBackwardMeta(
            backward_metas=self._stage_meta.input_grads or tuple()
        )

        if self.is_first:
            # First stage: no previous stage
            return None
        elif self._is_same_rank(self.stage_index - 1):
            # Same-rank: return for caller to pass
            return bwd_meta
        else:
            # Cross-rank: send via P2P
            self._send_meta(bwd_meta, self.stage_index - 1)
            return None

    def _prepare_backward_infra(
        self,
        num_microbatches: int,
        loss_fn: Callable[..., torch.Tensor] | None = None,
        target: torch.Tensor | None = None,
        received_grad_meta: "_StageBackwardMeta | None" = None,
    ) -> "_StageBackwardMeta | None":
        """
        Run backward metadata inference and prepare backward infrastructure.

        For V-shape schedules, multiple stages can be on the same rank.
        In this case, `received_grad_meta` contains metadata from the next
        stage (same rank), otherwise P2P receives it in `_backward_metadata_inference`.

        Args:
            num_microbatches: Number of microbatches
            loss_fn: Loss function for last stage
            target: Target tensor for last stage
            received_grad_meta: Gradient metadata from next stage (same-rank V-schedule)

        Returns:
            _StageBackwardMeta to pass to previous stage (same-rank or via P2P)
        """
        needs_inference = self._inference_mode == InferenceMode.DYNAMIC

        if needs_inference:
            # DYNAMIC mode: run backward metadata inference
            # received_grad_meta is used for same-rank V-schedule stages
            grad_meta_result = self._backward_metadata_inference(
                loss_fn=loss_fn,
                target=target,
                received_grad_meta=received_grad_meta,
            )
            # Validate dynamically inferred metadata against user-provided metadata
            self._validate_inferred_grad_metadata()
        else:
            # STATIC mode: metadata comes from user inputs, no validation needed
            grad_meta_result = _StageBackwardMeta(
                backward_metas=self._stage_meta.input_grads or ()
            )

        self._setup_backward_recv_info(num_microbatches)
        return grad_meta_result

    def _validate_inferred_grad_metadata(self) -> None:
        """Validate dynamically inferred grad metadata against user-provided metadata."""
        if self._user_meta is None:
            return
        if self._user_meta.input_grads and self._stage_meta.input_grads:
            validate_tensors_metadata(
                f"Stage {self.stage_index} input_grad",
                self._user_meta.input_grads,
                self._stage_meta.input_grads,
                warn_on_mismatch=True,
            )
        if self._user_meta.output_grads and self._stage_meta.output_grads:
            validate_tensors_metadata(
                f"Stage {self.stage_index} output_grad",
                self._user_meta.output_grads,
                self._stage_meta.output_grads,
                warn_on_mismatch=True,
            )

    def _prepare_forward_infra(
        self,
        num_microbatches: int,
        args: tuple[Any, ...] | _StageForwardMeta | None,
        kwargs: dict[str, Any] | None = None,
        has_backward: bool = False,
        loss_fn: Callable[..., torch.Tensor] | None = None,
        target: torch.Tensor | None = None,
    ) -> _StageForwardMeta | None:
        """
        Prepare the stage infrastructure for forward pass.

        For V-shape schedules, multiple stages can be on the same rank.
        In this case, `args` may be a `_StageForwardMeta` from the previous
        stage (same rank), otherwise P2P receives it in `_forward_metadata_inference`.

        Args:
            num_microbatches: Number of microbatches to prepare for
            args: Input arguments or _StageForwardMeta from prev stage (same-rank)
            kwargs: Keyword arguments for the stage
            has_backward: Whether backward pass will be performed
            loss_fn: Loss function (unused, kept for signature compatibility)
            target: Target tensor (unused, kept for signature compatibility)

        Returns:
            _StageForwardMeta to pass to next stage (same-rank or via P2P)
        """
        if self._inference_mode is None:
            raise RuntimeError(
                f"Stage {self.stage_index}: inference mode not set. "
                f"Call _determine_and_set_global_inference_mode() first."
            )

        fwd_meta_output: _StageForwardMeta | None = None
        needs_inference = (
            self._inference_mode == InferenceMode.DYNAMIC and self.inputs_meta is None
        )

        if needs_inference:
            # DYNAMIC mode: run forward metadata inference
            # args may be _StageForwardMeta for same-rank V-schedule stages
            fwd_meta_output = self._forward_metadata_inference(
                args, kwargs, has_backward
            )
            # Validate dynamically inferred metadata against user-provided metadata
            self._validate_inferred_forward_metadata()
        # STATIC mode: metadata comes from user inputs, no validation needed

        # Setup recv and send info
        self._setup_forward_recv_info(num_microbatches, has_backward)
        self._setup_forward_send_info()

        return fwd_meta_output

    def _validate_inferred_forward_metadata(self) -> None:
        """Validate dynamically inferred forward metadata against user-provided metadata."""
        if self._user_meta is None:
            return
        if self._user_meta.inputs and self._stage_meta.inputs:
            validate_tensors_metadata(
                f"Stage {self.stage_index} input",
                self._user_meta.inputs,
                self._stage_meta.inputs,
                warn_on_mismatch=True,
            )
        if self._user_meta.outputs and self._stage_meta.outputs:
            validate_tensors_metadata(
                f"Stage {self.stage_index} output",
                self._user_meta.outputs,
                self._stage_meta.outputs,
                warn_on_mismatch=True,
            )

    def _setup_forward_recv_info(
        self, num_microbatches: int, has_backward: bool
    ) -> None:
        """Setup receive info for forward pass."""
        for chunk_id in range(num_microbatches):
            if self.is_first:
                # First stage: all inputs are root arguments (no recv needed)
                self.args_recv_info[chunk_id] = tuple(
                    _RecvInfo(
                        input_name=f"root_input_{idx}",
                        source=None,
                        buffer=None,
                        tensor_meta=self._stage_meta.inputs[idx]
                        if self._stage_meta.inputs
                        else None,
                        is_root_arg=True,
                    )
                    for idx in range(len(self.inputs_meta or ()))
                )
            else:
                # Non-first stages: receive from previous stage
                if self._stage_meta.inputs is None:
                    raise PipeliningMetadataError(
                        f"Stage {self.stage_index}: inputs metadata required for recv info."
                    )

                recv_infos = tuple(
                    _RecvInfo(
                        f"recv_for_{self.stage_index}_from_{self.stage_index - 1}",
                        self.stage_index - 1,
                        _make_tensor_from_meta(meta, self.device),
                        meta,
                    )
                    for meta in self._stage_meta.inputs
                    if meta is not None
                )

                # Set requires_grad from metadata
                if has_backward:
                    for idx, r in enumerate(recv_infos):
                        if r.buffer is not None and self._stage_meta.inputs:
                            meta = (
                                self._stage_meta.inputs[idx]
                                if idx < len(self._stage_meta.inputs)
                                else None
                            )
                            if meta is not None:
                                r.buffer.requires_grad_(meta.requires_grad)

                self.args_recv_info[chunk_id] = recv_infos

    def _setup_forward_send_info(self) -> None:
        """Setup send info for forward pass."""
        self.act_send_info: dict[int, list] = {}
        if self._stage_meta.outputs is not None:
            for idx in range(len(self._stage_meta.outputs)):
                self.act_send_info[idx] = (
                    [self.stage_index + 1] if not self.is_last else []
                )

    def _create_grad_recv_info(
        self,
        act_send_info: dict,
    ) -> tuple[_RecvInfo, ...]:
        grad_recv_info: tuple[_RecvInfo, ...] = ()
        if not self.is_last:
            # Ensure output_grads metadata is available
            if self._stage_meta.output_grads is None:
                raise PipeliningMetadataError(
                    f"Stage {self.stage_index}: output_grads metadata is required for "
                    f"creating grad recv info. Ensure backward metadata is populated."
                )

            # Receiving gradients from multiple sources is not supported
            # hence we only take the first destination
            # Use a helper function to safely extract the metadata
            output_grads = self._stage_meta.output_grads
            grad_recv_info = tuple(
                _RecvInfo(
                    f"recv_grad_for_{self.stage_index}_from_{dst_list[0]}",
                    dst_list[0],
                    _make_tensor_from_meta(output_grads[idx], self.device),
                    output_grads[idx],
                )
                for idx, dst_list in act_send_info.items()
                if dst_list and output_grads[idx] is not None
            )
        return grad_recv_info
