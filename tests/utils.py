from torch import Tensor


def check_close(out: Tensor, out_ref: Tensor, *, rtol: float, atol: float, pct: float):
    out = out.float()
    out_ref = out_ref.float()

    tol = out_ref.abs() * rtol + atol
    diff = (out - out_ref).abs()
    mismatch = diff > tol
    mismatch_pct = mismatch.float().mean().item()

    msg = (
        f"mismatches: {mismatch_pct * 100:.2f}% < {pct * 100:.2f}%\n"
        f"largest absolute diff: {diff.max().item()}\n"
        f"largest relative diff: {(diff / out_ref.abs()).max().item()}"
    )
    assert mismatch_pct < pct, msg
