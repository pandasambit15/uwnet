import torch


def gradient_attack(inputs, eps):
    """Compute an adverserial example using the gradient of the loss

    Parameters
    ----------
    inputs: dict
        a dict of torch tensors with gradients computed
    eps: float
        magnitude of the attack

    Returns
    -------
    Perturbed example
    """
    out = {}
    for key, val in inputs.items():
        norm = torch.norm(val) / val.numel()
        norm_grad = val.grad * (eps / norm)
        norm_grad = torch.sign(val.grad) * eps
        attacked = val + norm_grad
        attacked = attacked.detach()
        attacked.requires_grad = True
        out[key] = attacked
    return out


def detach(inputs):
    out = {}
    for key in inputs:
        val = inputs[key].detach()
        val.requires_grad = True
        out[key] = val
    return out


def attacked_loss_and_gradients(closure, inputs, eps=.01, alpha=1.0):
    """Compute an attacked version of the loss function and the corresponding
    gradients"""

    inputs = detach(inputs)
    loss = closure(inputs)
    loss.backward()

    attacked_inputs = gradient_attack(inputs, eps)
    loss_attacked = alpha*closure(attacked_inputs)
    loss_attacked.backward()

    return loss + loss_attacked
