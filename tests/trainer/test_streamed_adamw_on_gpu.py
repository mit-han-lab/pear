import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the optimizer's states stream to a GPU")


def test_the_streamed_adamw_step_equals_adamw_bit_for_bit(monkeypatch):
    from pear.trainer.engine import StreamedAdamW
    from verl.utils.fsdp_utils import load_fsdp_optimizer, offload_fsdp_optimizer

    monkeypatch.setattr(StreamedAdamW, "CHUNK_BYTES", 64 << 10)  # several chunks through the three slots
    torch.manual_seed(7)
    sizes = (1000, 3000, 500, 7000, 20, 9000, 4000)
    offload_params = [torch.nn.Parameter(torch.randn(n, device="cuda")) for n in sizes]
    streamed_params = [torch.nn.Parameter(p.detach().clone()) for p in offload_params]
    offload_optimizer = torch.optim.AdamW(offload_params, lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1)
    streamed_optimizer = torch.optim.AdamW(streamed_params, lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1)
    stepper = StreamedAdamW(streamed_optimizer)

    for _ in range(4):
        for a, b in zip(offload_params, streamed_params, strict=True):
            a.grad = torch.randn_like(a)
            b.grad = a.grad.clone()
        # the engine's optimizer_offload_step: the offloaded states in for the update and out again; the streamed states
        # exist on the host from the start and stream from the first step
        load_fsdp_optimizer(offload_optimizer, "cuda")
        offload_optimizer.step()
        offload_fsdp_optimizer(offload_optimizer)
        assert stepper.is_ready()
        stepper.step()
        torch.cuda.synchronize()

    for a, b in zip(offload_params, streamed_params, strict=True):
        assert torch.equal(a, b)
        for key in ("exp_avg", "exp_avg_sq", "step"):
            assert torch.equal(offload_optimizer.state[a][key], streamed_optimizer.state[b][key])
