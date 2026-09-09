"""Inference checkpoints: geometry, cochains and the detached state used by the renderer.

These do not contain Adam or RNG state and are not training-resume checkpoints.
"""

import warnings

import torch


def save_state(model, run, step):
    state = {
        "step": int(step),
        "faces": model.faces_t.detach().cpu(),
        "x0": model.x0.detach().cpu(),
        "frame_mode": model.frame_mode,
        "aniso": float(model.aniso),
        "ref_e1": None if model.ref_e1 is None else model.ref_e1.detach().cpu(),
    }
    for name in ("verts", "sh_dc", "sh_rest", "opacity_logit"):
        state[name] = getattr(model, name).detach().cpu()
    torch.save(state, run / "state.pt")


def load_state(model, run):
    """Validate before copying; legacy runs reconstruct their unsaved reference frame."""
    state = torch.load(run / "state.pt", map_location="cpu", weights_only=True)
    if "verts" not in state:
        raise ValueError(f"{run}/state.pt has no vertices - not a deforming-mesh run")
    if "faces" in state and not torch.equal(state["faces"], model.faces_t.cpu()):
        raise ValueError("checkpoint connectivity differs from the initial mesh")
    if "x0" in state and not torch.equal(state["x0"], model.x0.cpu()):
        raise ValueError("checkpoint initial vertices differ from the initial mesh")
    for name in ("verts", "sh_dc", "sh_rest", "opacity_logit"):
        value = state[name]
        if value.shape != getattr(model, name).shape or not torch.isfinite(value).all():
            raise ValueError(f"invalid checkpoint tensor: {name}")
    mode = state.get("frame_mode", "shape_operator")
    aniso = float(state.get("aniso", 1.0))
    ref = state.get("ref_e1")
    if mode not in ("gram_schmidt", "shape_operator") or not 0 <= aniso <= 1:
        raise ValueError("invalid checkpoint frame mode or anisotropy")
    if ref is not None and (
        ref.shape != (len(model.faces_t), 3) or not torch.isfinite(ref).all()
    ):
        raise ValueError("invalid checkpoint reference frame")
    with torch.no_grad():
        for name in ("verts", "sh_dc", "sh_rest", "opacity_logit"):
            getattr(model, name).copy_(state[name])
    model.frame_mode, model.aniso = mode, aniso
    model.ref_e1 = None if ref is None else ref.to(model.faces_t.device)
    if "frame_mode" not in state:
        warnings.warn(
            "Legacy checkpoint: schedule and reference frame were not saved; "
            "assuming a completed ramp and rebuilding the reference. Render replay is approximate.",
            stacklevel=2,
        )
    if mode == "shape_operator" and model.ref_e1 is None:
        model.reassemble_reference()
    return int(state["step"])
