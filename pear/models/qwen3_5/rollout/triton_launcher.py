import torch


class TritonLauncher:
    """A Triton kernel launched through its compiled runner instead of the JIT."""

    DEFAULT_WARPS = 4  # warps per program unless a launch names another count (Triton's own default)

    def __init__(self, kernel):
        self.kernel = kernel
        self.compiled_kernels = {}  # key -> (compiled kernel, its constexpr names in signature order)
        self.device = None

    def __call__(self, key, grid, *args, num_warps=DEFAULT_WARPS, **constexprs):
        """Launch with the runtime arguments in signature order and every constexpr by name."""
        # the first launch of a key goes through the JIT, which compiles; the kernels mark their integer arguments
        # do_not_specialize and get 16-byte aligned pointers, so the caller's key (every value the compiled code
        # depends on) fixes the JIT's specialization
        entry = self.compiled_kernels.get(key)
        if entry is None:
            compiled_kernel = self.kernel[grid](*args, **constexprs, num_warps=num_warps)
            positions = {name: i for i, name in enumerate(self.kernel.arg_names)}
            self.compiled_kernels[key] = (compiled_kernel, sorted(constexprs, key=positions.__getitem__))
            return

        compiled_kernel, order = entry
        if self.device is None:
            self.device = torch.cuda.current_device()
        grid = (*grid, 1, 1)[:3]
        stream = torch._C._cuda_getCurrentRawStream(self.device)

        # the constexprs sit after the runtime arguments in these kernels
        compiled_kernel.run(
            *grid,
            stream,
            compiled_kernel.function,
            compiled_kernel.packed_metadata,
            None,
            None,
            None,
            *args,
            *(constexprs[name] for name in order),
        )
