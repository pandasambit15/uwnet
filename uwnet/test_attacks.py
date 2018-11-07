import torch
from toolz import first
import pytest
from .attacks import attacked_loss_and_gradients, gradient_attack


def get_first_item(x):
    arr = first(x.values())
    return arr[0]


def test_gradient_attack(eps=1.0, val=1.0):
    a = torch.tensor([val], requires_grad=True)
    input = {'a': a}

    loss = get_first_item(input)
    loss.backward()

    attacked = gradient_attack(input, eps=eps)
    expected = val + eps
    assert attacked['a'].item() == pytest.approx(expected)


def test_attacked_loss_and_gradients(val=1.0, eps=2.0, alpha=1.0):
    a = torch.tensor([val], requires_grad=True)
    input = {'a': a}
    loss = attacked_loss_and_gradients(get_first_item, input, eps, alpha)
    expected = val + alpha * (val + eps)
    assert loss.item() == pytest.approx(expected)
