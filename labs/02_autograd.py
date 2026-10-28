from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from labs.common import print_rule, resolve_device


def main() -> None:
    device = resolve_device()
    print("device:", device)

    print_rule("scalar chain rule")
    x = torch.tensor(2.0, device=device, requires_grad=True)
    w = torch.tensor(3.0, device=device, requires_grad=True)
    b = torch.tensor(-1.0, device=device, requires_grad=True)
    z = w * x + b
    y = z.square()
    y.backward()
    print("z, y:", z.item(), y.item())
    print("dw, db, dx:", w.grad.item(), b.grad.item(), x.grad.item())
    assert (z.item(), y.item()) == (5.0, 25.0)
    assert (w.grad.item(), b.grad.item(), x.grad.item()) == (20.0, 10.0, 30.0)

    print_rule("linear gradients")
    tokens, hidden, intermediate = 4, 3, 5
    X = torch.randn(tokens, hidden, device=device)
    W = torch.randn(intermediate, hidden, device=device)
    bias = torch.randn(intermediate, device=device)
    Y = X @ W.T + bias
    dY = torch.randn_like(Y)
    dW = dY.T @ X
    db = dY.sum(dim=0)
    dX = dY @ W
    print("Y,dW,db,dX:", tuple(Y.shape), tuple(dW.shape), tuple(db.shape), tuple(dX.shape))
    assert dW.shape == W.shape
    assert db.shape == bias.shape
    assert dX.shape == X.shape

    print_rule("SiLU gradient")
    points = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], device=device, requires_grad=True)
    F.silu(points).sum().backward()
    expected = torch.tensor(
        [-0.090784, 0.072329, 0.5, 0.927671, 1.090784],
        device=device,
    )
    print(points.grad)
    torch.testing.assert_close(points.grad, expected, rtol=1e-5, atol=1e-5)

    print_rule("gradient accumulation")
    p = torch.tensor(1.0, device=device, requires_grad=True)
    p.square().backward()
    p.square().backward()
    print("p.grad:", p.grad.item())
    assert p.grad.item() == 4.0

    print_rule("detach")
    a = torch.tensor(2.0, device=device, requires_grad=True)
    detached = (a * 3).detach()
    print("detached.requires_grad:", detached.requires_grad)
    assert not detached.requires_grad
    print("a.grad after detach-only computation:", a.grad)
    assert a.grad is None

    print_rule("gradcheck on CPU float64")
    cpu_x = torch.randn(7, dtype=torch.double, requires_grad=True)
    ok = torch.autograd.gradcheck(
        lambda t: F.silu(t).sum(),
        (cpu_x,),
        eps=1e-6,
        atol=1e-4,
    )
    print("gradcheck:", ok)
    assert ok

    print("\nPASS: chain rule, linear gradients, accumulation, detach and gradcheck")


if __name__ == "__main__":
    main()

