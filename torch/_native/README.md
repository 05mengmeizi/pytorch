# Native Ops (and DSLs)

The `torch._native` directory provides a place for PyTorch native ops written in python and DSLs, along with utilities to help facilitate this.

# Creating & Registering a native op

**All native ops must be registered to the dispatcher**

When writing native ops, they are required to interact meaningfully with torch's dispatcher, and thus must be registered correctly.

As a further clarification, ops cannot be labelled as `CompositeImplicitAutograd` in `native_functions.yaml`, as-in the op must have an explicit autograd function registered, or at minimum an explicit implementation registered for the same backend as being overridden/added.

## A Note on Imports

All registrations will happen at the end of `import torch`. It is expected at that point that **no DSL runtime library is loaded** - this means that the runtime(s) must only be imported lazily. We can still check the presence of a module, and get it's version without importing, but special care must be taken when writing op kernels to not import DSLs too early. An illustrative example is below, using `triton`:

First, we're going to write the registration function, and a top-level call, being very careful to not pull in the `triton` package early:

```
# torch/_native/ops/test_op/triton_impl.py

# NOTE: no triton import in the file

def calling_fn(...):
    # Lazily import the kernels (and triton) on first call
    from .triton_kernels import outer_fn
    torch.library.wrap_triton(outer_fn)(...)

def register_to_dispatch():
    torch.library.custom_op(...)(calling_fn)
    torch.library.register_autograd(...)
    torch.library.register_fake(...)

```

Note the lazy import of kernels inside `calling_fn` - this function isn't called during registration (it just needs to be defined), and on first call it'll import the `triton_kernels` module, with all its dependencies. This kernels module can import anything it wants internally, our restriction on no DSL imports during registration has been met. Quick example of `triton_kernels.py` below:

```
# torch/_native/ops/test_op/triton_kernels.py
import triton

@triton.jit
def inner_fn(...) -> ...:
    # depends internally on triton, triton.language
    pass

@triton.jit
def outer_fn(...) -> ...:
    # depends internally on triton, triton.language
    inner_fn(...)
```


## Registering Implementations to Existing Operators

There are 2 options when interacting with an existing operator:
1. Replace the operator for **all** cases with a new implementation
2. Replace **some subset of functionality** of a given operator with a new implementation, falling-back to the original implementation otherwise.

Both cases are very similar, with 2) only requiring an extra step to obtain the original implementation, and logic to determine which implementation should be run for a given case.

### Replacing an Operator

This follows a simple and standard path, with a good example being the implementation of [FlashAttention v4 (FAv4)](https://github.com/pytorch/pytorch/blob/1f66f34cda5b5ad02d231b90fa0c0de2cb4e02d1/torch/nn/attention/_fa4.py#L67) in torch.

The following example replaces the implementation of `aten._scaled_grouped_mm_v2` on `CUDA` devices:

```
# Note this must be global to avoid getting garbage collected
lib = None

def my_impl(...) -> ...:
    """
    Replacement implementation
    """
    pass

# Override the symbol `aten._scaled_grouped_mm_v2` in this example with the implementation in `my_impl`,
# noting the function signatures must match
def register_kernel_override():
    global lib

    # If already registered, don't do it again
    if lib is not None:
        return

    lib = torch.library.Library("aten", "IMPL", "CUDA")
    lib.impl("_scaled_grouped_mm_v2", my_impl, "CUDA")
```

### Replacing a Subset of Calls

This time we only want to override the behavior of a subset of `aten._scaled_grouped_mm_v2` calls, and choose whether to invoke our implementation or the original depending on some input arguments. Note that the core of the example -- creating a `torch.library.Library`, and registering our function using `lib.impl(...)` are the same as in [Replacing an Operator](#Replacing-an-Operator).

```
lib = None

def my_impl(...) -> ...:
    """
    Replacement implementation - laxy import actual implementation and run it
    """
    from .my_impl_kernel import my_kernel
    return my_kernel(...)

# Override the symbol `aten._scaled_grouped_mm_v2` in this example with the implementation in `my_impl`,
# only when the check-method `enable_my_impl` returns `True`
def register_kernel_override():
    global lib

    # If already registered, don't do it again
    if lib is not None:
        return

    # Get the original implementation for fallback purposes
    fallback_kernel = torch.library.get_kernel("aten::_scaled_grouped_mm_v2", "CUDA")

    # Note the dispatch_keys argument here - this must be passed as the first argument
    # to the fallback kernel
    def enable_my_impl(dispatch_keys, arg1, arg2, *args, **kwargs) -> bool:
        # determine if we want to call our implementation
        if arg1 == ... and arg2 == ...:
            return my_impl(arg1, arg2, *args, **kwargs)
        else:
            # Call the fallback
            return fallback_kernel(dispatch_keys,
                                   arg1, arg2, *args, **kwargs)

    # Same as before
    lib = torch.library.Library("aten", "IMPL", "CUDA")
    # Pass the enablement function, note needed with_keyset=True argument to
    # get and pass dispatch keys
    lib.impl("_scaled_grouped_mm_v2", enable_my_impl, "CUDA", with_keyset=True)
```

## Registering a New Operator

We currently don't have any use for this, please come talk to us if you want this!

# Adding a new DSL

Adding a new DSL is as simple as adding a single helper utils file, then writing your op.

Some universal utilities are provided in `common_utils.py`:
* `check_native_jit_disabled() -> bool` : returns `True` if native DSL ops have been globally disabled.
* `_available_version(package: str) -> tuple[int,int,int] | None` : Gets the installed version of `package` without importing it
* `_unavailable_reasons(deps: list[tuple[str,str]]) -> str | None` : For a list of `(package_name, module_name)` pairs, check (without importing) if the module is available, returning a string describing the mitigation step if it isn't.

## DSL utils file spec

A DSL utils file, named `$dsl_utils.py` (i.e. `cutedsl_utils.py` for `$dsl=cutedsl`) requires three methods to be implemented.

1. `runtime_available() -> bool` : tell the user if the runtime is available - note that this needs to be available during init, and must be fork-safe. Packages should also not be imported at this time - rely on `importlib.util.find_spec(package_name)` or similar to get the necessary information without importing.
2. `runtime_version() -> tuple[int, int, int]` : return the `(major, minor, update)` version of the installed package.
3. `register_op(Callable[..., None])` : Register a given op, where `Callable[..., None]` is similar to those defined in [Creating and registering...]. This can contain DSL-specific checks / features, for example one might choose to only register ops only if the DSL version is one of a pre-approved list.

An example of an implementation of this spec can be found in [cutedsl_utils.py](cutedsl_utils.py), but please talk to us if you're planning on adding a new DSL.
