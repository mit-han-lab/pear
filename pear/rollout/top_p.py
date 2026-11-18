# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Changed from vLLM 0.34.0's vllm/v1/sample/ops/topk_topp_triton.py: _topk_topp_kernel's standalone top-p path
# alone, its softmax shifted by each row's exact max instead of its first block's mean + 10 std.
from vllm.triton_utils import tl, triton
from vllm.v1.sample.ops import topk_topp_triton
from vllm.v1.sample.ops.topk_topp_triton import _update_min_larger_stats


@triton.jit(do_not_specialize_on_alignment=["BATCH_SIZE"])
def _exact_max_top_p_kernel(
    LOGITS,
    LOGITS_STRIDE_0,
    BUFFER,
    NORMAL_CDF_TO_SIGMA_TABLE,
    P,
    BATCH_SIZE,
    VOCAB_SIZE: tl.constexpr,
    MASK_VALUE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_TRUNC: tl.constexpr,
):
    # vLLM's _topk_topp_kernel without top-k: a program per row in turn, the row's top-p mask in place
    NUM_TILES: tl.constexpr = (VOCAB_SIZE + BLOCK_SIZE - 1) // BLOCK_SIZE
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    for row_id in tl.range(pid, BATCH_SIZE, num_programs):
        LOGITS_ROW = LOGITS + row_id.to(tl.int64) * LOGITS_STRIDE_0
        BUFFER_ROW = BUFFER + pid * VOCAB_SIZE

        final_pivot = -float("inf")
        duplicate_logit = float("inf")
        num_duplicate_logit = tl.zeros((), dtype=tl.uint32)
        num_keep = tl.zeros((), dtype=tl.uint32)
        num_kept = tl.zeros((), dtype=tl.uint32)

        max_logit = -float("inf")
        min_logit = float("inf")

        p = tl.load(P + row_id)
        if p < 1.0:
            # Zeroth pass: Compute avg and std from a sample block
            offs = tl.arange(0, BLOCK_SIZE)
            mask_n = offs < VOCAB_SIZE
            logits_blk0 = tl.load(LOGITS_ROW + offs, mask=mask_n, other=-float("inf"))
            # Exclude -inf values (e.g. from grammar bitmasks) from
            # statistics to avoid NaN in pivot computation.
            finite_mask = (logits_blk0 > -float("inf")) & mask_n
            num_finite = tl.sum(finite_mask)
            finite_logits = tl.where(finite_mask, logits_blk0, 0.0)
            avg_logit = tl.where(num_finite > 0, tl.sum(finite_logits) / num_finite, 0.0)
            sq_avg_logit = tl.where(
                num_finite > 0,
                tl.sum(finite_logits * finite_logits) / num_finite,
                0.0,
            )
            std_logit = tl.sqrt(tl.maximum(sq_avg_logit - avg_logit * avg_logit, 0.0))
            # the row's exact max: exp(x - max_sample) <= 1, no overflow, no probability above the search's sentinel 1.0
            max_sample = -float("inf")
            for i in range(0, NUM_TILES):
                offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                logits_blk = tl.load(LOGITS_ROW + offs_n, mask=offs_n < VOCAB_SIZE, other=-float("inf"))
                max_sample = tl.maximum(max_sample, tl.max(logits_blk))
            sum_exp_logits = 0.0

            # First pass: compute max and min logits and sum_exp_logits
            for i in range(0, NUM_TILES):
                offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                mask_n = offs_n < VOCAB_SIZE
                logits_blk = tl.load(LOGITS_ROW + offs_n, mask=mask_n, other=-float("inf"))
                max_logit = tl.maximum(max_logit, tl.max(logits_blk))
                # Exclude -inf from min to keep binary search bounds
                # finite (avoids NaN pivots).
                finite_blk = tl.where(logits_blk > -float("inf"), logits_blk, float("inf"))
                min_logit = tl.minimum(min_logit, tl.min(finite_blk))

                probs_blk = tl.exp(logits_blk - max_sample)
                probs_blk = tl.where(mask_n, probs_blk, 0.0)
                sum_exp_logits += tl.sum(probs_blk)

            # If no finite logits exist (all -inf), clamp min to
            # max so the search converges to -inf (no masking).
            min_logit = tl.minimum(min_logit, max_logit)

            idx = tl.cast(p * 200, tl.int32)
            idx = tl.maximum(0, tl.minimum(idx, 199))
            sigma = tl.load(NORMAL_CDF_TO_SIGMA_TABLE + idx)
            sigma = sigma + tl.abs(sigma) * -0.25
            outlier_pivot = avg_logit + std_logit * sigma

            outlier_prob = tl.exp(outlier_pivot - max_sample) / sum_exp_logits
            sum_outlier_probs = 0.0
            num_outliers = tl.zeros((), dtype=tl.uint32)

            # Second pass: Calculate softmax and gather outliers
            for i in range(0, NUM_TILES):
                offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                mask_n = offs_n < VOCAB_SIZE

                probs_blk = tl.load(LOGITS_ROW + offs_n, mask=mask_n, other=-float("inf"))
                probs_blk = tl.exp(probs_blk - max_sample)
                probs_blk = probs_blk / sum_exp_logits

                outlier_mask = (probs_blk > outlier_prob) & mask_n
                sum_outlier_probs += tl.sum(outlier_mask * probs_blk)
                cumulative_pos = tl.cast(tl.cumsum(outlier_mask) - 1 + num_outliers, tl.int32)
                num_outliers += tl.sum(outlier_mask)
                write_pos = tl.where(outlier_mask, cumulative_pos, -1)
                tl.store(BUFFER_ROW + write_pos, probs_blk, mask=outlier_mask)

            max_range = tl.exp(max_logit - max_sample) / sum_exp_logits
            min_range = tl.exp(min_logit - max_sample) / sum_exp_logits

            p_pivot = 1.0
            num_iters = 0
            min_larger_prob = 1.0
            num_min_larger = tl.zeros((), dtype=tl.uint32)
            p_pivots_sum = 0.0

            # Third pass: Search for p_pivot
            if sum_outlier_probs > p:
                min_range = outlier_prob
                search_range = tl.cast(num_outliers, tl.int32)
                search_iters = tl.cast(
                    (num_outliers + BLOCK_SIZE_TRUNC - 1) // BLOCK_SIZE_TRUNC,
                    tl.int32,
                )

                found_pivot = 0
                while found_pivot == 0:
                    p_pivot_0 = (max_range - min_range) * 0.5 + min_range
                    p_pivots_sum_0 = 0.0
                    min_larger_0 = 1.0
                    num_min_larger_0 = tl.zeros((), dtype=tl.uint32)

                    # Single fused pass: compute p_pivots_sum,
                    # min_larger, and num_min_larger together.
                    # See _update_min_larger_stats for the
                    # tile-level merge logic.
                    for i in range(0, search_iters):
                        offs_n = i * BLOCK_SIZE_TRUNC + tl.arange(0, BLOCK_SIZE_TRUNC)
                        mask_n_2 = offs_n < search_range
                        probs_blk = tl.load(BUFFER_ROW + offs_n, mask=mask_n_2, other=0.0)

                        above_0 = probs_blk > p_pivot_0
                        p_pivots_sum_0 += tl.sum(probs_blk * above_0)

                        min_larger_0, num_min_larger_0 = _update_min_larger_stats(
                            probs_blk,
                            above_0,
                            min_larger_0,
                            num_min_larger_0,
                            1.0,
                        )

                    # Check if the pivot satisfies termination condition
                    if p_pivots_sum_0 >= p and p_pivots_sum_0 - (min_larger_0 * num_min_larger_0) < p:
                        p_pivot = p_pivot_0
                        min_larger_prob = min_larger_0
                        num_min_larger = num_min_larger_0
                        p_pivots_sum = p_pivots_sum_0
                        found_pivot = 1

                    # Update range
                    if p_pivots_sum_0 > p:
                        min_range = p_pivot_0
                    elif p_pivots_sum_0 < p:
                        max_range = p_pivot_0

                    num_iters += 1
                    if (max_range - min_range) < 1e-9 or num_iters >= 18:
                        p_pivot = (max_range + min_range) / 2.0
                        min_larger_prob = min_larger_0
                        num_min_larger = num_min_larger_0
                        p_pivots_sum = p_pivots_sum_0
                        found_pivot = 1
            else:
                # Re-populate the buffer with full softmax probabilities
                for i in range(0, NUM_TILES):
                    offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                    mask_n = offs_n < VOCAB_SIZE

                    probs_blk = tl.load(LOGITS_ROW + offs_n, mask=mask_n, other=-float("inf"))
                    probs_blk = tl.exp(probs_blk - max_sample)
                    probs_blk = probs_blk / sum_exp_logits
                    tl.store(BUFFER_ROW + offs_n, probs_blk, mask=mask_n)

                found_pivot = 0
                while found_pivot == 0:
                    p_pivot_0 = (max_range - min_range) * 0.5 + min_range
                    p_pivots_sum_0 = 0.0
                    min_larger_0 = 1.0
                    num_min_larger_0 = tl.zeros((), dtype=tl.uint32)

                    # Single fused pass: compute p_pivots_sum,
                    # min_larger, and num_min_larger together.
                    # See _update_min_larger_stats for the
                    # tile-level merge logic.
                    for i in range(0, NUM_TILES):
                        offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                        mask_n = offs_n < VOCAB_SIZE
                        probs_blk = tl.load(BUFFER_ROW + offs_n, mask=mask_n, other=0.0)

                        above_0 = probs_blk > p_pivot_0
                        p_pivots_sum_0 += tl.sum(probs_blk * above_0)

                        min_larger_0, num_min_larger_0 = _update_min_larger_stats(
                            probs_blk,
                            above_0,
                            min_larger_0,
                            num_min_larger_0,
                            1.0,
                        )

                    # Check if the pivot satisfies termination condition
                    if p_pivots_sum_0 >= p and p_pivots_sum_0 - (min_larger_0 * num_min_larger_0) < p:
                        p_pivot = p_pivot_0
                        min_larger_prob = min_larger_0
                        num_min_larger = num_min_larger_0
                        p_pivots_sum = p_pivots_sum_0
                        found_pivot = 1

                    # Update range
                    if p_pivots_sum_0 > p:
                        min_range = p_pivot_0
                    elif p_pivots_sum_0 < p:
                        max_range = p_pivot_0

                    num_iters += 1
                    if (max_range - min_range) < 1e-9 or num_iters >= 18:
                        p_pivot = (max_range + min_range) / 2.0
                        min_larger_prob = min_larger_0
                        num_min_larger = num_min_larger_0
                        p_pivots_sum = p_pivots_sum_0
                        found_pivot = 1

            duplicate_logit = tl.log(min_larger_prob * sum_exp_logits) + max_sample
            num_duplicate_logit = num_min_larger
            num_keep = num_duplicate_logit - tl.cast((p_pivots_sum - p) / min_larger_prob, tl.uint32)
            num_kept = tl.zeros((), dtype=tl.uint32)

            # Top-p only path
            final_pivot = tl.log(p_pivot * sum_exp_logits) + max_sample

        # Sixth pass: Apply mask and store final output.
        # If the pivot >= max logit (or is NaN), no token would
        # survive the strict `>` keep_mask.  Skip masking.
        # Using `not <` instead of `>=` so that NaN is also caught.
        if not (final_pivot < max_logit):
            final_pivot = -float("inf")
        elif final_pivot != -float("inf"):
            for i in range(0, NUM_TILES):
                offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                mask_n = offs_n < VOCAB_SIZE
                logits_blk = tl.load(LOGITS_ROW + offs_n, mask=mask_n, other=-float("inf"))
                keep_mask = (logits_blk > final_pivot) & mask_n

                # Duplicate logit handling
                if num_keep < num_duplicate_logit:
                    duplicate_mask = (tl.abs(logits_blk - duplicate_logit) < 1e-9) & mask_n
                    duplicate_count = tl.cumsum(duplicate_mask) + num_kept
                    duplicate_keep_mask = (duplicate_count <= num_keep) & duplicate_mask
                    duplicate_remove_mask = duplicate_mask & ~duplicate_keep_mask
                    num_kept += tl.sum(duplicate_keep_mask)
                    keep_mask = keep_mask & (~duplicate_remove_mask)

                logits_blk = tl.where(keep_mask, logits_blk, MASK_VALUE)
                tl.store(LOGITS_ROW + offs_n, logits_blk, mask=mask_n)


class ExactMaxTopP:
    """vLLM 0.34's Triton top-p of a batch without top-k, its softmax shifted by each row's exact max: launched where
    vLLM launches its own kernel, vLLM's dispatch unchanged (its split pipeline below 65 rows)."""

    # vLLM's launch of the kernel on CUDA: the tiles a program sweeps a row in, its search's tiles, its warps
    BLOCK_SIZE = 8192
    BLOCK_SIZE_TRUNC = 4096
    NUM_WARPS = 8

    vllm_launch = None  # vLLM's _topk_topp, for batches with top-k

    @classmethod
    def install(cls) -> None:
        """Take vLLM's launch of its top-k/top-p kernel, once per process."""
        # vLLM's engine warmup (register_top_k_top_p_warmups) compiles vLLM's own kernels only: this one compiles at
        # its first launch in a worker
        if cls.vllm_launch is None:
            cls.vllm_launch = topk_topp_triton._topk_topp
            topk_topp_triton._topk_topp = cls.launch

    @classmethod
    def launch(cls, logits, buffer, percentile_to_std_table, normal_cdf_to_sigma_table, k, p, mask_value, num_sm):
        """vLLM's _topk_topp: a batch without top-k on this kernel, any other on vLLM's."""
        if k is not None or p is None:
            return cls.vllm_launch(
                logits, buffer, percentile_to_std_table, normal_cdf_to_sigma_table, k, p, mask_value, num_sm
            )

        batch_size, vocab_size = logits.shape
        _exact_max_top_p_kernel[(min(num_sm, batch_size),)](
            logits,
            logits.stride(0),
            buffer,
            normal_cdf_to_sigma_table,
            p,
            batch_size,
            VOCAB_SIZE=vocab_size,
            MASK_VALUE=mask_value,
            BLOCK_SIZE=cls.BLOCK_SIZE,
            BLOCK_SIZE_TRUNC=cls.BLOCK_SIZE_TRUNC,
            num_warps=cls.NUM_WARPS,
        )
