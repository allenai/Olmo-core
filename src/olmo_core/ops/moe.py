from typing import Any, List, Optional, Tuple, Union

import torch
import torch.distributed as dist

try:
    from olmo_core.kernels import moe as kernels
except (ImportError, RuntimeError):
    kernels = None  # type: ignore


def _is_eligible(x):
    return x.is_floating_point() and x.is_cuda and (x.dtype is not torch.float64)


def _cast(x, dtype):
    if isinstance(x, torch.Tensor) and _is_eligible(x):
        return x.to(dtype)
    elif isinstance(x, dict):
        return {_cast(k, dtype): _cast(v, dtype) for k, v in x.items()}
    elif isinstance(x, (list, tuple)):
        return type(x)(map(lambda y: _cast(y, dtype), x))
    return x


class GatherOp(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx: Any,
        x: torch.Tensor,
        indices: torch.Tensor,
        bin_ids: torch.Tensor,
        bins: torch.Tensor,
        top_k: int,
    ):
        assert kernels is not None
        ctx.save_for_backward(indices, bin_ids, bins)
        ctx.top_k = top_k
        return kernels.gather(x, indices, bin_ids, None, bins, top_k)

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx: Any, grad: torch.Tensor):
        assert kernels is not None
        grad = grad.contiguous()
        indices, bin_ids, bins = ctx.saved_tensors
        out = kernels.scatter(grad, indices, bin_ids, None, bins, ctx.top_k)
        return out, None, None, None, None, None


def gather(
    x: torch.Tensor,
    indices: torch.Tensor,
    bin_ids: torch.Tensor,
    bins: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    return GatherOp.apply(x, indices, bin_ids, bins, top_k)  # type: ignore


class ScatterOp(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx: Any,
        x: torch.Tensor,
        indices: torch.Tensor,
        bin_ids: torch.Tensor,
        weights: Optional[torch.Tensor],
        bins: torch.Tensor,
        top_k: int,
    ) -> torch.Tensor:
        assert kernels is not None
        maybe_x = [x] if ctx.needs_input_grad[3] else []
        ctx.save_for_backward(indices, bin_ids, weights, bins, *maybe_x)
        ctx.top_k = top_k
        ctx.x_shape = x.shape
        return kernels.scatter(x, indices, bin_ids, weights, bins, top_k)

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx: Any, grad: torch.Tensor):
        assert kernels is not None

        grad = grad.contiguous()
        saved_tensors = ctx.saved_tensors

        indices, bin_ids, weights, bins = saved_tensors[:4]
        dgrad = None
        if ctx.needs_input_grad[0]:
            dgrad = kernels.gather(
                grad,
                indices,
                bin_ids,
                weights,
                bins,
                ctx.top_k,
            )

        wgrad = None
        if ctx.needs_input_grad[3]:  # need wgrad
            x = saved_tensors[-1]
            wgrad = kernels.scatter_wgrad(
                x,
                grad,
                indices,
                bin_ids,
                bins,
                ctx.top_k,
            )
        return dgrad, None, None, wgrad, None, None, None


def scatter(
    x: torch.Tensor,
    indices: torch.Tensor,
    bin_ids: torch.Tensor,
    weights: Optional[torch.Tensor],
    bins: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    return ScatterOp.apply(x, indices, bin_ids, weights, bins, top_k)  # type: ignore


class BinnedGatherOp(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx: Any,
        x: torch.Tensor,
        indices: torch.Tensor,
        bins: torch.Tensor,
        bin_size: int,
        top_k: int,
    ):
        assert kernels is not None
        ctx.save_for_backward(indices, bins)
        ctx.top_k = top_k
        return kernels.binned_gather(x, indices, None, bins, bin_size, top_k)

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx: Any, grad: torch.Tensor):
        assert kernels is not None
        grad = grad.contiguous()
        indices, bins = ctx.saved_tensors
        out = kernels.binned_scatter(grad, indices, None, bins, ctx.top_k)
        return out, None, None, None, None


def binned_gather(
    x: torch.Tensor, indices: torch.Tensor, bins: torch.Tensor, bin_size: int, top_k: int
) -> torch.Tensor:
    return BinnedGatherOp.apply(x, indices, bins, bin_size, top_k)  # type: ignore


class BinnedScatterOp(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx: Any,
        x: torch.Tensor,
        indices: torch.Tensor,
        weights: Optional[torch.Tensor],
        bins: torch.Tensor,
        top_k: int,
    ):
        assert kernels is not None

        assert len(x.size()) == 3
        ctx.bin_size = x.size(1)
        ctx.top_k = top_k

        # TODO: Don't save 'x' for backwards if we don't need to
        # calculate the gradient w.r.t. 'weights'.
        ctx.save_for_backward(x, indices, weights, bins)
        return kernels.binned_scatter(x, indices, weights, bins, top_k)

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx: Any, grad: torch.Tensor):
        assert kernels is not None

        grad = grad.contiguous()
        x, indices, weights, bins = ctx.saved_tensors
        out = kernels.binned_gather(
            grad,
            indices,
            weights,
            bins,
            ctx.bin_size,
            ctx.top_k,
        )

        wgrad = None
        if ctx.needs_input_grad[2]:
            wgrad = kernels.binned_scatter_wgrad(
                x,
                grad,
                indices,
                bins,
                ctx.top_k,
            )
        return out, None, wgrad, None, None


def binned_scatter(
    x: torch.Tensor,
    indices: torch.Tensor,
    weights: Optional[torch.Tensor],
    bins: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    return BinnedScatterOp.apply(x, indices, weights, bins, top_k)  # type: ignore


def repeat(x: torch.Tensor, tiling: Union[torch.Size, Tuple[int, ...]]) -> torch.Tensor:
    if all((t == 1 for t in tiling)):
        return x
    return x.repeat(*tiling)


class AllToAllOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, output_split_sizes, input_split_sizes, group, async_op):
        if output_split_sizes is not None:
            out = torch.empty(
                (sum(output_split_sizes),) + x.shape[1:], device=x.device, dtype=x.dtype
            )
        else:
            out = torch.empty_like(x)

        ctx.input_shape = x.shape
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes
        ctx.group = group
        handle = dist.all_to_all_single(
            out,
            x,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
            async_op=async_op,
        )
        return out, handle

    @staticmethod
    def backward(ctx, grad, _):
        if ctx.needs_input_grad[0]:
            out = torch.empty(
                ctx.input_shape,
                device=grad.device,
                dtype=grad.dtype,
            )
            dist.all_to_all_single(
                out,
                grad,
                output_split_sizes=ctx.input_split_sizes,
                input_split_sizes=ctx.output_split_sizes,
                group=ctx.group,
            )
            return out, None, None, None, None
        return None, None, None, None, None


def all_to_all(
    x: torch.Tensor,
    output_split_sizes: Optional[List[int]] = None,
    input_split_sizes: Optional[List[int]] = None,
    group: Optional[dist.ProcessGroup] = None,
    async_op: bool = False,
) -> Tuple[torch.Tensor, dist.Work]:
    return AllToAllOp.apply(  # type: ignore
        x,
        output_split_sizes,
        input_split_sizes,
        group,
        async_op,
    )


class AllToAllAsyncOp(torch.autograd.Function):
    """
    Autograd function backing :func:`all_to_all_async` / :func:`all_to_all_wait`.

    :func:`all_to_all_async` launches the all-to-all and returns immediately so the
    caller can overlap unrelated compute with the communication; :func:`all_to_all_wait`
    later blocks on the handle. The work handle and the metadata needed by the wait /
    backward passes are stashed as attributes on the output tensor (``_a2a_*``) so they
    travel through autograd without extra ``ctx`` plumbing across the two ops.

    The backward mirrors the forward, so the gradient all-to-all overlaps too:
    ``AllToAllWaitOp.backward`` launches the reverse all-to-all and ``AllToAllAsyncOp.backward``
    waits on it. The in-flight handle is handed between them by stashing it on the gradient of
    the passthrough ``x``.

    That handoff only works if ``x`` has a single consumer (the matching
    :func:`all_to_all_wait`), because Python attributes don't survive tensor arithmetic.
    Tracing the gradient of ``x``:

    - One consumer (correct)::

        WaitOp.backward -> x_grad (._a2a_handle attached)
                        -> AsyncOp.backward gets that same tensor -> handle found, waited

    - Two consumers / fan-out (wrong)::

        WaitOp.backward -> grad_A (._a2a_handle attached)
        the other op    -> grad_B (no attribute)
        autograd sums   -> grad_A + grad_B = a NEW tensor (handle lost)
                        -> AsyncOp.backward gets the new tensor -> handle missing -> raises

    The guard in ``AllToAllAsyncOp.backward`` turns the fan-out case into a clear error instead
    of a silent desync. The single-consumer contract is not otherwise enforced.
    """

    @staticmethod
    def forward(ctx, x, output_split_sizes, input_split_sizes, group):
        if output_split_sizes is not None:
            y = torch.empty(
                (sum(output_split_sizes),) + x.shape[1:], device=x.device, dtype=x.dtype
            )
        else:
            y = torch.empty_like(x)

        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes
        ctx.group = group
        y_handle = dist.all_to_all_single(
            y,
            x.contiguous(),
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
            async_op=True,
        )
        # Carry the metadata the matching AllToAllWaitOp needs on the output tensor.
        setattr(y, "_a2a_input_shape", x.shape)
        setattr(y, "_a2a_output_split_sizes", output_split_sizes)
        setattr(y, "_a2a_input_split_sizes", input_split_sizes)
        setattr(y, "_a2a_group", group)

        return x, y, y_handle

    @staticmethod
    def backward(ctx, *grad_outputs):
        x_grad = grad_outputs[0]
        # The reverse all-to-all handle was stashed on this gradient by AllToAllWaitOp.backward.
        # If it's missing, the passthrough was fanned out to more than one consumer: autograd
        # summed the gradients into a fresh tensor, which dropped the attribute (see class doc).
        handle = getattr(x_grad, "_a2a_handle", None)
        if handle is None:
            raise RuntimeError(
                "all_to_all_async backward could not find its in-flight gradient handle. The "
                "passthrough tensor returned by all_to_all_async must feed ONLY the matching "
                "all_to_all_wait; fanning it out to other ops makes autograd accumulate "
                "gradients into a new tensor that drops the stashed handle."
            )
        handle.wait()
        del x_grad._a2a_handle
        return x_grad, None, None, None, None


class AllToAllWaitOp(torch.autograd.Function):
    """Completion half of :class:`AllToAllAsyncOp` (see :func:`all_to_all_wait`)."""

    @staticmethod
    def forward(ctx, x, y, y_handle):
        # y is the output tensor from AllToAllAsyncOp; read the stashed input shape
        # rather than y.shape (which is the post-all-to-all shape).
        ctx.input_shape = y._a2a_input_shape
        ctx.output_split_sizes = y._a2a_output_split_sizes
        ctx.input_split_sizes = y._a2a_input_split_sizes
        ctx.group = y._a2a_group
        del y._a2a_output_split_sizes
        del y._a2a_input_split_sizes
        del y._a2a_group
        del y._a2a_input_shape

        y_handle.wait()
        return y

    @staticmethod
    def backward(ctx, *grad_outputs):
        y_grad = grad_outputs[0]
        x_grad = torch.empty(
            ctx.input_shape,
            device=y_grad.device,
            dtype=y_grad.dtype,
        )
        x_grad_handle = dist.all_to_all_single(
            x_grad,
            y_grad.contiguous(),
            output_split_sizes=ctx.input_split_sizes,
            input_split_sizes=ctx.output_split_sizes,
            group=ctx.group,
            async_op=True,
        )
        # Stash the in-flight handle on the gradient; AllToAllAsyncOp.backward waits on it.
        setattr(x_grad, "_a2a_handle", x_grad_handle)
        return x_grad, y_grad, None


@torch._dynamo.disable()
def all_to_all_async(
    x: torch.Tensor,
    output_split_sizes: Optional[List[int]] = None,
    input_split_sizes: Optional[List[int]] = None,
    group: Optional[dist.ProcessGroup] = None,
) -> Tuple[torch.Tensor, torch.Tensor, dist.Work]:
    """
    Launch a non-blocking all-to-all and return without waiting on it.

    Pairs with :func:`all_to_all_wait` to overlap communication with compute::

        x, y, handle = all_to_all_async(x, ...)
        z = some_independent_compute(w)   # overlaps; note: NOT on x (see warning)
        y = all_to_all_wait(x, y, handle) # blocks until the transfer is done

    :param x: The local input tensor (sharded along dim 0).
    :param output_split_sizes: Per-rank receive counts; ``None`` for an even split.
    :param input_split_sizes: Per-rank send counts; ``None`` for an even split.
    :param group: The process group to communicate over.

    :returns: ``(x, y, handle)`` — the (unchanged) input ``x``, the output buffer ``y``
        (not yet valid until waited on), and the in-flight work ``handle``.

    .. warning::
        The returned passthrough ``x`` must feed **only** the matching
        :func:`all_to_all_wait` — do not fan it out to other ops. The backward all-to-all
        handle is stashed as an attribute on ``x``'s gradient and carried along that single
        autograd edge from :class:`AllToAllWaitOp` to :class:`AllToAllAsyncOp`. If ``x`` has
        more than one consumer, autograd sums the incoming gradients into a fresh tensor that
        drops the stashed handle, breaking the backward pass. This contract is not enforced.

    .. note::
        The *only* thing this op adds over the existing ``all_to_all(..., async_op=True)``
        (which already returns a work handle, giving forward comm/compute overlap) is
        overlap of the **backward** gradient all-to-all, via the paired-``Function`` design.
        So it pays off only when ``backward()`` runs. The synchronous EP strategies that use
        it are, in practice, mostly the **eval/inference** path (no-sync configs fall back to
        them at eval because no-sync can hang on uneven per-rank token counts) — and eval runs
        no backward, so there this is equivalent to a plain async ``all_to_all`` but with the
        passthrough footgun above. Production *training* uses the no-sync (symmetric-memory)
        path, which doesn't use this op at all; only a few sync-training debug/ablation runs
        actually exercise the backward overlap.

        TODO: revisit whether this op earns its keep. If the sync EP path can use
        ``all_to_all(..., async_op=True)`` for its forward overlap, consider dropping
        ``all_to_all_async`` / ``all_to_all_wait`` (and their passthrough footgun) entirely.
    """
    return AllToAllAsyncOp.apply(  # type: ignore
        x,
        output_split_sizes,
        input_split_sizes,
        group,
    )


@torch._dynamo.disable()
def all_to_all_wait(x: torch.Tensor, y: torch.Tensor, y_handle: dist.Work) -> torch.Tensor:
    """
    Block until the all-to-all launched by :func:`all_to_all_async` completes.

    :param x: The original input tensor returned by :func:`all_to_all_async` (kept in the
        autograd graph so the backward all-to-all is wired correctly).
    :param y: The output buffer returned by :func:`all_to_all_async`.
    :param y_handle: The work handle returned by :func:`all_to_all_async`.

    :returns: The completed output tensor ``y``.
    """
    return AllToAllWaitOp.apply(x, y, y_handle)  # type: ignore


def sum_tensor(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    if x.shape[dim] == 1:
        return x.squeeze(dim=dim)
    return x.sum(dim=dim)


def batched_histc(x: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    A batched version of ``torch.histc``.
    """
    hist = torch.zeros((*x.shape[:-1], num_classes), dtype=x.dtype, device=x.device)
    ones = torch.ones_like(x)
    hist.scatter_add_(-1, x, ones)
    return hist


def histc(x: torch.Tensor, num_classes: int) -> torch.Tensor:
    # NOTE: 'torch.histc' not implemented for integers on CPU, so convert to float then back to ints on CPU.
    if x.device.type == "cpu":
        return torch.histc(x.float(), bins=num_classes, min=0, max=num_classes - 1).int()
    else:
        return torch.histc(x, bins=num_classes, min=0, max=num_classes - 1)


def segment_ids_from_eos(input_ids: torch.Tensor, eos_token_id: int) -> torch.Tensor:
    """Return per-token document IDs, with each EOS token starting the next segment."""
    eos = (input_ids == eos_token_id).to(torch.long)
    eos[:, 0] = 0
    return eos.cumsum(dim=1)


def segment_ids_from_doc_lens(doc_lens: torch.Tensor, seq_len: int) -> torch.Tensor:
    """Return per-token document IDs from per-instance document lengths.

    ``doc_lens`` has shape ``(batch_size, max_docs)``, zero-padded on the right, and each row must
    sum to ``seq_len``, as the collator produces for intra-document masking. IDs are contiguous,
    nondecreasing and zero-based within each sequence, like :func:`segment_ids_from_eos`.
    """
    ends = torch.cumsum(doc_lens.to(torch.long), dim=1)
    if not bool((ends[:, -1] == seq_len).all()):
        raise ValueError(f"doc_lens must sum to the sequence length ({seq_len}) in every row")
    starts = torch.zeros(doc_lens.shape[0], seq_len + 1, dtype=torch.long, device=doc_lens.device)
    # Each document end is the next document's start; `scatter_` (not add) keeps IDs contiguous
    # across zero-length entries, whose ends repeat.
    starts.scatter_(1, ends, 1)
    return starts[:, :seq_len].cumsum(dim=1)


def doc_sum_scatter(per_token: torch.Tensor, segment_ids: torch.Tensor) -> torch.Tensor:
    """Sum values within each document and broadcast the sums back to its tokens."""
    batch_size, seq_len, num_experts = per_token.shape
    indices = segment_ids.unsqueeze(-1).expand(-1, -1, num_experts)
    document_sums = torch.zeros(
        batch_size,
        seq_len,
        num_experts,
        dtype=per_token.dtype,
        device=per_token.device,
    )
    document_sums.scatter_add_(1, indices, per_token)
    return document_sums.gather(1, indices)


def pool_keep_mask(
    document_scores: torch.Tensor,
    pool_size_per_token: torch.Tensor,
) -> torch.Tensor:
    """Return a mask retaining each document's highest-scoring expert pool."""
    ranks = document_scores.argsort(dim=-1, descending=True).argsort(dim=-1)
    return ranks < pool_size_per_token.unsqueeze(-1)


def pool_keep_mask_inverse_scatter(
    document_scores: torch.Tensor,
    pool_size_per_token: torch.Tensor,
) -> torch.Tensor:
    """Invert the same sorting permutation with scatter instead of a second sort.

    This preserves the first argsort's tie policy exactly. Each destination is written
    once because ``order`` is a permutation, so there are no scatter write collisions.
    Kept separate until controlled profiling qualifies the alternative implementation.
    """
    order = document_scores.argsort(dim=-1, descending=True)
    positions = torch.arange(order.shape[-1], device=order.device, dtype=order.dtype)
    ranks = torch.empty_like(order)
    ranks.scatter_(-1, order, positions.expand_as(order))
    return ranks < pool_size_per_token.unsqueeze(-1)
