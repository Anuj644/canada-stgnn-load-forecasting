"""Replay small optimizer steps on CUDA without changing the minibatch objective."""
from __future__ import annotations

import torch


class TensorPreservingPlateau(torch.optim.lr_scheduler.ReduceLROnPlateau):
    """Keep captured optimizer learning-rate storage alive during reductions.

    PyTorch 2.6's parent scheduler replaces a tensor LR with a Python float.
    Captured kernels still reference the original tensor, so reductions must
    update that same storage. Float-based eager optimizers retain parent behavior.
    """
    def _reduce_lr(self, epoch):
        previous = [group["lr"] for group in self.optimizer.param_groups]
        super()._reduce_lr(epoch)
        for group, original in zip(self.optimizer.param_groups, previous):
            if isinstance(original, torch.Tensor):
                original.fill_(float(group["lr"]))
                group["lr"] = original


class CapturedStep:
    """Capture a fixed-size Adam step, padding only the final batch with zero weight.

    loss_builder(indices, mask) must use CUDA-resident constants exclusively.
    Model weights and Adam moments are restored after warmup/capture. Learning
    rate is a CUDA scalar tensor so plateau scheduling remains effective.
    """
    def __init__(self, model, optimizer, loss_builder, batch_size, device):
        rng_state = torch.cuda.get_rng_state()
        self.ids = torch.arange(batch_size, device=device)
        self.mask = torch.ones(batch_size, device=device)
        self.batch_size = batch_size
        params = [p for p in model.parameters() if p.requires_grad]
        weights = [p.detach().clone() for p in params]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                optimizer.zero_grad(set_to_none=True)
                loss_builder(self.ids, self.mask).backward()
        torch.cuda.current_stream().wait_stream(stream)
        for p in params:
            state = optimizer.state[p]
            if not state:
                state["step"] = torch.zeros((), dtype=torch.float32, device=device)
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
        moments = {p: {key: value.clone() for key, value in optimizer.state[p].items()}
                   for p in params}
        optimizer.zero_grad(set_to_none=True)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.loss = loss_builder(self.ids, self.mask)
            self.loss.backward()
            optimizer.step()
        with torch.no_grad():
            for p, weight in zip(params, weights):
                p.copy_(weight)
                for key, value in optimizer.state[p].items():
                    if isinstance(value, torch.Tensor):
                        value.copy_(moments[p][key])
        torch.cuda.set_rng_state(rng_state)

    def __call__(self, ids):
        count = len(ids)
        self.ids[:count].copy_(ids)
        self.mask.fill_(1)
        if count != self.batch_size:
            self.ids[count:].zero_()
            self.mask[count:].zero_()
        self.graph.replay()
        return self.loss.detach()


class CapturedSequenceStep:
    """Replay several sequential minibatches; each still performs its own update.

    This is not gradient accumulation or a larger optimization batch. It reduces
    host launches, which are expensive on Windows. Remainders use ordinary eager
    steps with the same gradients and Adam moments.
    """
    def __init__(self, model, optimizer, loss_builder, batch_size, steps, device, initial_ids=None):
        rng_state = torch.cuda.get_rng_state()
        self.size = batch_size * steps
        self.ids = (torch.arange(self.size, device=device) if initial_ids is None
                    else initial_ids[:self.size].to(device=device, dtype=torch.long).clone())
        if len(self.ids) != self.size:
            raise ValueError("Capture warmup needs a complete group of training indices")
        self.mask = torch.ones(batch_size, device=device)
        self.loss = torch.zeros((), device=device)
        params = [p for p in model.parameters() if p.requires_grad]
        weights = [p.detach().clone() for p in params]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                optimizer.zero_grad(set_to_none=False)
                loss_builder(self.ids[:batch_size], self.mask).backward()
        torch.cuda.current_stream().wait_stream(stream)
        for p in params:
            if not optimizer.state[p]:
                optimizer.state[p].update(step=torch.zeros((), device=device),
                                          exp_avg=torch.zeros_like(p), exp_avg_sq=torch.zeros_like(p))
        moments = {p: {key: value.clone() for key, value in optimizer.state[p].items()} for p in params}
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.loss.zero_()
            for step in range(steps):
                optimizer.zero_grad(set_to_none=False)
                loss = loss_builder(self.ids[step * batch_size:(step + 1) * batch_size], self.mask)
                loss.backward()
                optimizer.step()
                self.loss.add_(loss.detach() / steps)
        with torch.no_grad():
            for p, weight in zip(params, weights):
                p.copy_(weight)
                for key, value in optimizer.state[p].items():
                    value.copy_(moments[p][key])
        torch.cuda.set_rng_state(rng_state)

    def __call__(self, ids):
        if len(ids) != self.size:
            raise ValueError("Only complete groups are replayed; use eager steps for remainders")
        self.ids.copy_(ids)
        self.graph.replay()
        return self.loss.detach()
