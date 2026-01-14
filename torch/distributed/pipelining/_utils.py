# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, TYPE_CHECKING

import torch
from torch import fx
from torch.distributed._mesh_layout import _MeshLayout
from torch.distributed.tensor import DTensor


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.tensor.placement_types import Placement


logger = logging.getLogger(__name__)


# Key for mesh cache: (mesh_dim_names, mesh_layout)
# mesh_layout is the _MeshLayout object containing shape and stride (not actual ranks).
# This uniquely identifies a mesh within the same "universe" where all stages share
# the same rank tensor.
MeshCacheKey = tuple[tuple[str, ...], Optional[_MeshLayout]]


class PipeliningMetadataError(RuntimeError):
    """
    Metadata mismatch errors in pipeline communication.

    This is the unified exception for all metadata-related errors including:
    - Shape/dtype/stride mismatches
    - DTensor placement/mesh mismatches
    - Metadata validation failures
    """


@dataclass(frozen=True)
class _TensorMeta:
    """
    Base metadata for tensors used in pipeline parallelism.

    This dataclass captures the essential properties of a tensor needed for
    recv buffer allocation and tensor validation during pipeline communication.

    For plain tensors: These are the tensor's actual shape/stride.
    For DTensors: These are LOCAL tensor attributes (the shard's shape/stride),
                  used for recv buffer allocation. Global attributes are stored
                  separately in _DTensorMeta.

    Attributes:
        shape: The shape of the tensor (local shard shape for DTensors).
        stride: The stride of the tensor (local shard stride for DTensors).
        dtype: The data type of the tensor.
        requires_grad: Whether the tensor requires gradient computation.
    """

    shape: torch.Size
    stride: tuple[int, ...]
    dtype: torch.dtype
    requires_grad: bool

    @staticmethod
    def from_tensor(tensor: torch.Tensor) -> _TensorMeta:
        """Extract metadata from a plain tensor."""
        if isinstance(tensor, DTensor):
            raise TypeError(
                "Expected plain tensor, got DTensor. Use _DTensorMeta.from_dtensor instead."
            )
        return _TensorMeta(
            shape=tensor.shape,
            stride=tensor.stride(),
            dtype=tensor.dtype,
            requires_grad=tensor.requires_grad,
        )

    def to_meta_tensor(self) -> torch.Tensor:
        """Create a meta tensor from this metadata."""
        return torch.empty_strided(
            self.shape,
            self.stride,
            dtype=self.dtype,
            device="meta",
            requires_grad=self.requires_grad,
        )

    def get_diff(self, other: _TensorMeta) -> list[str]:
        """
        Get list of differences between this metadata and another.

        Returns empty list if equal, otherwise list of difference descriptions.
        Uses dataclass __eq__ for equality check first as fast path.
        """
        if self == other:
            return []

        diffs = []
        if self.shape != other.shape:
            diffs.append(f"shape: {self.shape} vs {other.shape}")
        if self.stride != other.stride:
            diffs.append(f"stride: {self.stride} vs {other.stride}")
        if self.dtype != other.dtype:
            diffs.append(f"dtype: {self.dtype} vs {other.dtype}")
        if self.requires_grad != other.requires_grad:
            diffs.append(
                f"requires_grad: {self.requires_grad} vs {other.requires_grad}"
            )
        return diffs


@dataclass(frozen=True)
class _DTensorMeta(_TensorMeta):
    """
    Metadata needed to reconstruct a DTensor from a local tensor.

    Extends _TensorMeta with DTensor-specific distribution information.

    This dataclass captures all information required to reconstruct a DTensor
    on a receiving pipeline stage, including mesh identification for multi-mesh
    scenarios where different DTensors may live on different submeshes within
    a mesh universe.

    This dataclass is frozen (immutable) and contains only serializable fields.
    The actual DeviceMesh object is NOT stored here because it cannot be
    serialized for P2P transmission. Instead, the mesh is cached separately
    in PipelineStage._mesh_cache and looked up using (mesh_dim_names, mesh_layout)
    as the cache key.

    Attribute inheritance:
        - shape, stride, dtype, requires_grad: LOCAL tensor attributes (from _TensorMeta)
          Used for recv buffer allocation and local tensor validation.
        - global_shape, global_stride: GLOBAL DTensor attributes
          Used for DTensor reconstruction via DTensor.from_local().
    """

    # Global DTensor properties (for reconstruction)
    global_shape: torch.Size = field(default_factory=lambda: torch.Size([]))
    global_stride: tuple[int, ...] = field(default=())

    # DTensor distribution properties
    placements: tuple[Placement, ...] = field(
        default=()
    )  # e.g., (Shard(0), Replicate())

    # Mesh identification - used to look up the correct DeviceMesh from cache
    mesh_dim_names: tuple[str, ...] = field(default=())  # e.g., ("tp",) or ("dp", "tp")
    mesh_layout: _MeshLayout | None = field(
        default=None
    )  # _MeshLayout with shape/stride - uniquely identifies mesh within the same universe

    @staticmethod
    def from_dtensor(dtensor: DTensor) -> _DTensorMeta:
        """Extract metadata from a DTensor for P2P transmission."""
        spec = dtensor._spec
        device_mesh = dtensor.device_mesh
        local_tensor = dtensor.to_local()

        return _DTensorMeta(
            # Local tensor attributes (for recv buffer allocation)
            shape=local_tensor.shape,
            stride=local_tensor.stride(),
            dtype=dtensor.dtype,
            requires_grad=dtensor.requires_grad,
            # Global DTensor attributes (for reconstruction)
            global_shape=spec.shape,
            global_stride=spec.stride,
            # Distribution info
            placements=spec.placements,
            mesh_dim_names=(
                tuple(device_mesh.mesh_dim_names) if device_mesh.mesh_dim_names else ()
            ),
            mesh_layout=device_mesh._layout,
        )

    @property
    def mesh_cache_key(self) -> MeshCacheKey:
        """Get the cache key for mesh lookup."""
        return (self.mesh_dim_names, self.mesh_layout)

    def to_meta_dtensor(self, mesh: DeviceMesh) -> DTensor:
        """
        Create a meta DTensor from this metadata.

        Unlike to_meta_tensor() which returns a plain tensor, this method
        reconstructs the full DTensor with placements. This is essential for
        forward metadata inference when the model contains DTensor operations.

        Args:
            mesh: The DeviceMesh to use for the DTensor.

        Returns:
            A DTensor with meta local tensor and proper placements.
        """
        local_meta = torch.empty_strided(
            self.shape,
            self.stride,
            dtype=self.dtype,
            device="meta",
            requires_grad=self.requires_grad,
        )
        return DTensor.from_local(
            local_meta,
            device_mesh=mesh,
            placements=self.placements,
            shape=self.global_shape,
            stride=self.global_stride,
            run_check=False,
        )

    def get_diff(self, other: _TensorMeta) -> list[str]:
        """
        Get list of differences between this DTensor metadata and another.

        Extends base class comparison with DTensor-specific fields.
        """
        if self == other:
            return []

        # Get base class differences (compares local shape/stride/dtype/requires_grad)
        diffs = super().get_diff(other)

        # Add DTensor-specific comparisons if other is also _DTensorMeta
        if isinstance(other, _DTensorMeta):
            if self.global_shape != other.global_shape:
                diffs.append(
                    f"global_shape: {self.global_shape} vs {other.global_shape}"
                )
            if self.global_stride != other.global_stride:
                diffs.append(
                    f"global_stride: {self.global_stride} vs {other.global_stride}"
                )
            if self.placements != other.placements:
                diffs.append(f"placements: {self.placements} vs {other.placements}")
            if self.mesh_dim_names != other.mesh_dim_names:
                diffs.append(
                    f"mesh_dim_names: {self.mesh_dim_names} vs {other.mesh_dim_names}"
                )
            if self.mesh_layout != other.mesh_layout:
                diffs.append(f"mesh_layout: {self.mesh_layout} vs {other.mesh_layout}")
        else:
            diffs.append("type: _DTensorMeta vs _TensorMeta")

        return diffs


# Type alias for union of tensor metadata types
TensorMeta = _TensorMeta | _DTensorMeta


@dataclass
class _StageMeta:
    """
    All tensor metadata for a pipeline stage.

    Consolidates input/output metadata for both forward and backward passes.
    """

    inputs: tuple[_TensorMeta | None, ...] | None = None
    outputs: tuple[_TensorMeta | None, ...] | None = None
    input_grads: tuple[_TensorMeta | None, ...] | None = None
    output_grads: tuple[_TensorMeta | None, ...] | None = None

    def has_any(self) -> bool:
        """Check if any metadata field is populated."""
        return any(
            v is not None
            for v in [self.inputs, self.outputs, self.input_grads, self.output_grads]
        )

    def has_dtensors(self) -> bool:
        """Check if any input/output metadata is DTensor type."""
        for metas in [self.inputs, self.outputs]:
            if metas and any(isinstance(m, _DTensorMeta) for m in metas if m):
                return True
        return False

    def is_complete_for_forward(self) -> bool:
        """Check if forward metadata is fully populated."""
        return self.inputs is not None and self.outputs is not None


@dataclass(frozen=True)
class _StageForwardMeta:
    """
    Forward metadata transmitted from stage i to stage i+1 during metadata inference.

    Contains metadata about stage outputs which become inputs for the next stage.
    """

    forward_metas: tuple[
        TensorMeta | None, ...
    ]  # Stage i's outputs → Stage i+1's inputs


@dataclass(frozen=True)
class _StageBackwardMeta:
    """
    Backward metadata transmitted from stage i to stage i-1 during backward metadata inference.

    Contains metadata about input gradients which become output gradients for the previous stage.
    The gradient metadata allows receiving stages to properly reconstruct DTensor gradients
    with the correct placements (which may differ from forward activation placements,
    e.g., Replicate → Partial).
    """

    backward_metas: tuple[
        TensorMeta | None, ...
    ]  # Stage i's input_grads → Stage i-1's output_grads


def _make_tensor_from_meta(
    meta: _TensorMeta,
    device: torch.device,
) -> torch.Tensor:
    """
    Create a recv buffer tensor from tensor metadata.

    Uses torch.empty_strided to preserve the exact memory layout of the original tensor,
    which is important for proper P2P communication of strided tensors and DTensor locals.

    Args:
        meta: Tensor metadata containing shape, stride, and dtype
        device: Device to allocate the tensor on

    Returns:
        Empty tensor with the specified shape, stride, dtype on the target device
    """
    return torch.empty_strided(
        size=meta.shape,
        stride=meta.stride,
        dtype=meta.dtype,
        device=device,
    )


class _MeshCache:
    """
    Cache for DeviceMesh objects keyed by (mesh_dim_names, mesh_layout).

    mesh_layout is the _MeshLayout object (shape + stride) which uniquely identifies
    a mesh within the same "universe". This assumes all pipeline stages share the same
    rank tensor, which is true for TorchTitan and similar frameworks where meshes are
    derived from a common world.

    This centralizes mesh lookup logic that was previously scattered in PipelineStage.
    The cache allows reconstruction of DTensors on receiving stages without requiring
    a callback to create/lookup meshes.
    """

    def __init__(self) -> None:
        self._cache: dict[MeshCacheKey, DeviceMesh] = {}

    def get(self, key: MeshCacheKey) -> DeviceMesh | None:
        """Get a mesh by cache key, or None if not found."""
        return self._cache.get(key)

    def get_or_create(
        self,
        key: MeshCacheKey,
        get_mesh_callback: Callable[
            [tuple[str, ...], Optional[_MeshLayout]], DeviceMesh
        ]
        | None = None,
    ) -> DeviceMesh:
        """
        Get a mesh by key, creating it via callback if not cached.

        Args:
            key: The cache key (mesh_dim_names, mesh_layout).
            get_mesh_callback: Optional callback to create mesh if not cached.
                Takes (mesh_dim_names, mesh_layout) and returns a DeviceMesh.
                mesh_layout is a _MeshLayout object with shape and stride.

        Returns:
            The DeviceMesh.

        Raises:
            PipeliningMetadataError: If mesh not found and no callback provided,
                or if callback returns None.
        """
        if key in self._cache:
            return self._cache[key]

        mesh_dim_names, mesh_layout = key

        if get_mesh_callback is None:
            raise PipeliningMetadataError(
                f"Mesh not found in cache for mesh_dim_names={mesh_dim_names}, "
                f"mesh_layout={mesh_layout}, and no get_mesh callback provided. "
                f"Provide a get_mesh callback or use DTensors in static mode."
            )

        mesh = get_mesh_callback(mesh_dim_names, mesh_layout)
        if mesh is None:
            raise PipeliningMetadataError(
                f"Mesh lookup failed: callback returned None for "
                f"mesh_dim_names={mesh_dim_names}, mesh_layout={mesh_layout}. "
                f"Ensure all stages use meshes from the same universe."
            )
        self._cache[key] = mesh
        return mesh

    def put(self, key: MeshCacheKey, mesh: DeviceMesh) -> None:
        """Add a mesh to the cache."""
        self._cache[key] = mesh

    def update_from_tensors(self, tensors: tuple[torch.Tensor | None, ...]) -> None:
        """
        Extract and cache meshes from DTensors in a tensor tuple.

        Args:
            tensors: Tuple of tensors, which may include DTensors.
        """
        for tensor in tensors:
            if isinstance(tensor, DTensor):
                mesh = tensor.device_mesh
                dim_names = tuple(mesh.mesh_dim_names) if mesh.mesh_dim_names else ()
                mesh_layout = mesh._layout
                key = (dim_names, mesh_layout)
                if key not in self._cache:
                    self._cache[key] = mesh

    def __contains__(self, key: MeshCacheKey) -> bool:
        return key in self._cache

    def __len__(self) -> int:
        return len(self._cache)


# ============================================================================
# Inference mode enum
# ============================================================================


class InferenceMode(Enum):
    """
    Enum representing the metadata inference mode for pipeline parallelism.

    This is a **pipeline-level** property determined collectively by all stages
    across all PP ranks during schedule initialization. The mode is determined
    by the schedule (not individual stages) because:
    1. `has_backward` is only known at schedule creation time
    2. All stages must agree on the mode to avoid P2P communication hangs

    Modes:
        STATIC: All stages have sufficient metadata to skip runtime inference.
            This includes:
            - input_args + output_args + input_grads + output_grads (full metadata)
            - input_args + output_args with all plain tensors (grad shapes derivable)
            - input_args + output_args with DTensors but has_backward=False (no grads needed)

        DYNAMIC: At least one stage requires runtime metadata inference.
            This includes:
            - No args provided (full runtime inference)
            - Partial args provided (only input_args or output_args)
            - DTensors present with has_backward=True but missing grad metadata

    The mode is determined via a collective (all-reduce) across PP ranks to ensure
    consistency. If ANY stage needs DYNAMIC, all stages use DYNAMIC.
    """

    STATIC = "static"
    DYNAMIC = "dynamic"

    @classmethod
    def needs_dynamic(cls, meta: _StageMeta, has_backward: bool) -> bool:
        """
        Determine if dynamic metadata inference is needed.

        This is the core logic for mode determination, used by schedules to
        collectively decide the global inference mode across all PP ranks.

        Args:
            meta: Stage metadata extracted from user-provided args.
            has_backward: Whether backward pass will be performed.

        Returns:
            True if dynamic inference is needed, False if static is sufficient.
        """
        # Case 1: Forward metadata incomplete → needs DYNAMIC
        if not meta.is_complete_for_forward():
            return True

        # Case 2: No DTensors → STATIC is fine (grad shapes derivable from tensor shapes)
        if not meta.has_dtensors():
            return False

        # Case 3: No backward needed → STATIC is fine (don't need grad metadata)
        if not has_backward:
            return False

        # Case 4: DTensors with backward but missing ANY grad metadata → needs DYNAMIC
        # Both input_grads AND output_grads are required for static mode with DTensors
        if meta.input_grads is None or meta.output_grads is None:
            return True

        # Case 5: DTensors with complete grads → STATIC is fine
        return False


# ============================================================================
# Utility functions
# ============================================================================


def flatten_args(args, *, detach: bool = False):
    """
    Flatten the args into a list form.

    Args:
        args: The arguments to flatten.
        detach: If True, detach tensors from computational graph while preserving requires_grad.

    Returns:
        If detach=True: (new_args, flat_detached_args) where new_args has same structure with detached tensors.
        If detach=False: flat_args list.
    """
    flat_args = []

    if detach:

        def extract_and_detach(a):
            nonlocal flat_args
            if isinstance(a, torch.Tensor):
                val = a.detach().requires_grad_(a.requires_grad)
                flat_args.append(val)
                return val
            else:
                flat_args.append(a)
                return a

        new_args = fx.node.map_aggregate(args, extract_and_detach)
        return new_args, flat_args
    else:

        def extract_only(a):
            nonlocal flat_args
            flat_args.append(a)
            return a

        fx.node.map_aggregate(args, extract_only)
        return flat_args


# Backward compatibility alias
def flatten_args_detach(args):
    """
    Flatten the args into a list form and detach the tensors from computational graph.

    Deprecated: Use flatten_args(args, detach=True) instead.
    """
    return flatten_args(args, detach=True)


def generate_stage_to_rank_mapping(
    pp_size: int, num_stages: int, style: str = "loop"
) -> dict[int, int]:
    """
    Compute the stage id to rank mapping for either a looped or V-style schedule.

    Most commonly num_stages == pp_size * 2, but this function can be used to
    compute the mapping for any number of stages per rank.
    """
    mapping = {}
    if style == "loop":
        for stage_index in range(num_stages):
            mapping[stage_index] = stage_index % pp_size
    elif style == "v":
        if num_stages % pp_size != 0:
            raise ValueError(
                f"num_stages {num_stages} must be evenly divisible by pp_size {pp_size} for V schedules"
            )

        rank_index = 0
        for stage_index in range(num_stages):
            mapping[stage_index] = rank_index
            # dont change rank if we are on the border (to keep v shape)
            if (stage_index + 1) % pp_size == 0:
                continue
            if (stage_index // pp_size) % 2 == 0:
                rank_index += 1
            else:
                rank_index -= 1
    else:
        raise ValueError(f"Style {style} is not supported.")
    return mapping


def generate_rank_to_stage_mapping(
    pp_size: int, num_stages: int, style: str = "loop"
) -> dict[int, list[int]]:
    """
    Compute the rank to stage id mapping for either a looped or V-style schedule.

    This function inverts the stage_to_rank_mapping to get which stages are assigned to each rank.

    Returns a dictionary mapping rank -> list of stage indices assigned to that rank.
    """
    stage_to_rank = generate_stage_to_rank_mapping(pp_size, num_stages, style)

    # Invert the mapping: rank -> list of stages
    rank_to_stages: dict[int, list[int]] = {}
    for stage_id, rank in stage_to_rank.items():
        if rank not in rank_to_stages:
            rank_to_stages[rank] = []
        rank_to_stages[rank].append(stage_id)

    # Sort the stage lists for each rank to ensure consistent ordering
    for stages in rank_to_stages.values():
        stages.sort()

    return rank_to_stages


@dataclass
class PipeInfo:
    """
    Captures information for a pipeline (`Pipe` object).
    """

    graph: fx.Graph
    num_stages: int
    has_loss_and_backward: bool


# ============================================================================
# Metadata extraction helpers
# ============================================================================


def extract_tensor_meta(tensor: torch.Tensor) -> TensorMeta:
    """
    Extract metadata from a tensor (either plain tensor or DTensor).

    Args:
        tensor: The tensor to extract metadata from.

    Returns:
        _TensorMeta for plain tensors, _DTensorMeta for DTensors.
    """
    if isinstance(tensor, DTensor):
        return _DTensorMeta.from_dtensor(tensor)
    else:
        return _TensorMeta.from_tensor(tensor)


def extract_tensor_metas(
    tensors: tuple[torch.Tensor | None, ...],
) -> tuple[TensorMeta | None, ...]:
    """
    Extract metadata from a tuple of tensors.
    Args:
        tensors: Tuple of tensors (may include None).

    Returns:
        Tuple of metadata objects (_TensorMeta or _DTensorMeta), with None preserved.
    """
    return tuple(extract_tensor_meta(t) if t is not None else None for t in tensors)


def to_local_if_dtensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert DTensor to local tensor, or return tensor unchanged if already local.

    This utility simplifies the common pattern of extracting local tensors:
        local = tensor.to_local() if isinstance(tensor, DTensor) else tensor

    Args:
        tensor: A tensor that may be a DTensor or plain tensor.

    Returns:
        The local tensor component.
    """
    if isinstance(tensor, DTensor):
        return tensor.to_local()
    return tensor


def normalize_to_tuple(
    args: torch.Tensor | tuple[torch.Tensor, ...] | None,
) -> tuple[torch.Tensor, ...] | None:
    """
    Normalize input args to a tuple.

    Args:
        args: A single tensor, tuple of tensors, or None.

    Returns:
        Tuple of tensors, or None if input is None.
    """
    if args is None:
        return None
    if isinstance(args, (torch.Tensor, DTensor)):
        return (args,)
    return args


# ============================================================================
# Validation functions
# ============================================================================


def validate_metadata(
    desc: str,
    expected: TensorMeta,
    actual: torch.Tensor | TensorMeta,
    *,
    raise_on_mismatch: bool = False,
    warn_on_mismatch: bool = False,
) -> list[str]:
    """
    Compare expected metadata against actual tensor or metadata.

    This is the unified validation/comparison function that uses get_diff() from
    metadata classes. Works with both plain tensors and DTensors.

    For plain tensors: compares shape/stride/dtype/requires_grad.
    For DTensors: compares all properties including global shape and placements.

    Args:
        desc: Description for error/warning messages.
        expected: Expected tensor metadata (_TensorMeta or _DTensorMeta).
        actual: Actual tensor or metadata to compare against.
        raise_on_mismatch: If True, raise PipeliningMetadataError on mismatch.
        warn_on_mismatch: If True, issue a warning on mismatch.

    Returns:
        List of differences (empty if metadata matches).

    Raises:
        PipeliningMetadataError: If raise_on_mismatch=True and differences exist.
    """
    # Extract metadata if actual is a tensor
    if isinstance(actual, torch.Tensor):
        actual_meta = extract_tensor_meta(actual)
    else:
        actual_meta = actual

    # Type check: ensure both are same type for meaningful comparison
    if type(expected) is not type(actual_meta):
        type_diff = [
            f"type: expected {type(expected).__name__}, got {type(actual_meta).__name__}"
        ]
        if raise_on_mismatch:
            raise PipeliningMetadataError(f"{desc}: {type_diff[0]}")
        if warn_on_mismatch:
            warnings.warn(
                f"{desc}: Metadata type mismatch. {type_diff[0]}. "
                f"Using dynamically inferred metadata instead.",
                UserWarning,
            )
        return type_diff

    # Use get_diff() from the metadata class
    diffs = expected.get_diff(actual_meta)

    if diffs:
        if raise_on_mismatch:
            raise PipeliningMetadataError(f"{desc}: {'; '.join(diffs)}")
        if warn_on_mismatch:
            warnings.warn(
                f"{desc}: Metadata mismatch. {'; '.join(diffs)}. "
                f"Using dynamically inferred metadata instead.",
                UserWarning,
            )

    return diffs


def validate_tensors_metadata(
    desc: str,
    expected_metas: tuple[TensorMeta | None, ...],
    actuals: tuple[torch.Tensor | TensorMeta | None, ...],
    *,
    skip_none_actuals: bool = False,
    warn_on_mismatch: bool = False,
) -> None:
    """
    Validate that a collection of actual tensors/metadata match expected metadata.
    This is a convenience wrapper for validating lists/tuples of tensors or metadata.

    Args:
        desc: Description for error messages.
        expected_metas: Expected tensor metadata (tuple of _TensorMeta/_DTensorMeta or None).
        actuals: Actual tensors or metadata to validate.
        skip_none_actuals: If True, skip validation for None actual tensors. This is useful
            for backward pass validation where gradients can legitimately be None
            for non-differentiable outputs.
        warn_on_mismatch: If True, issue warnings instead of raising errors.

    Raises:
        PipeliningMetadataError: If lengths don't match or any tensor metadata mismatches
            (unless warn_on_mismatch=True).
    """
    if len(expected_metas) != len(actuals):
        msg = (
            f"{desc}: Number of values ({len(actuals)}) does not match "
            f"expected number ({len(expected_metas)})"
        )
        if warn_on_mismatch:
            warnings.warn(f"{msg}. Using dynamic metadata.", UserWarning)
            return
        raise PipeliningMetadataError(msg)

    for i, (expected, actual) in enumerate(zip(expected_metas, actuals, strict=True)):
        # Handle None cases
        if expected is None and actual is None:
            continue
        if expected is None:
            msg = f"{desc} [{i}]: expected None, got {type(actual).__name__}"
            if warn_on_mismatch:
                warnings.warn(f"{msg}. Using dynamic metadata.", UserWarning)
                continue
            raise PipeliningMetadataError(msg)
        if actual is None:
            if skip_none_actuals:
                continue
            msg = f"{desc} [{i}]: expected {type(expected).__name__}, got None"
            if warn_on_mismatch:
                warnings.warn(f"{msg}. Using dynamic metadata.", UserWarning)
                continue
            raise PipeliningMetadataError(msg)

        validate_metadata(
            f"{desc} [{i}]",
            expected,
            actual,
            warn_on_mismatch=warn_on_mismatch,
            raise_on_mismatch=not warn_on_mismatch,
        )


def validate_static_dtensor_grad_correspondence(
    stage_index: int,
    args: tuple[torch.Tensor, ...],
    grads: tuple[torch.Tensor | None, ...],
    is_input: bool,
) -> None:
    """
    Validate DTensor↔grad correspondence for static mode.

    For each DTensor in args with requires_grad=True, the corresponding grad
    must also be a DTensor. For non-DTensor args or args with requires_grad=False,
    the corresponding grad can be None or a plain tensor.

    Args:
        stage_index: The stage index for error messages.
        args: Tuple of forward tensors.
        grads: Tuple of gradient tensors (can include None).
        is_input: True for input_args/input_grads, False for output_args/output_grads.

    Raises:
        ValueError: If tuple lengths don't match or DTensor↔grad correspondence fails.
    """
    kind = "input" if is_input else "output"
    args_name = f"{kind}_args"
    grads_name = f"{kind}_grads"

    if len(args) != len(grads):
        raise ValueError(
            f"Stage {stage_index}: {grads_name} length ({len(grads)}) does not match "
            f"{args_name} length ({len(args)}). Each forward tensor must have a "
            f"corresponding gradient entry (use None for tensors that don't require grad)."
        )

    for i, (arg, grad) in enumerate(zip(args, grads, strict=True)):
        if isinstance(arg, DTensor) and arg.requires_grad:
            if grad is None:
                raise ValueError(
                    f"Stage {stage_index}: {args_name}[{i}] is a DTensor with requires_grad=True, "
                    f"but {grads_name}[{i}] is None. For DTensors requiring grad, "
                    f"provide a DTensor gradient with the expected placement."
                )
            if not isinstance(grad, DTensor):
                raise ValueError(
                    f"Stage {stage_index}: {args_name}[{i}] is a DTensor with requires_grad=True, "
                    f"but {grads_name}[{i}] is {type(grad).__name__}, expected DTensor. "
                    f"DTensor gradients may have different placements than forward tensors."
                )
