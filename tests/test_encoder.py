import pytest
import torch
from src.models.encoder import GridEncoder
from src.models.action_head import ActionHead
from src.models.policy_net import PolicyNetwork
from src.models.value_net import ValueNetwork


class TestGridEncoder:
    def test_output_shape(self):
        enc = GridEncoder(16, 256)
        x = torch.randn(4, 16, 64, 64)
        out = enc(x)
        assert out.shape == (4, 256)

    def test_single_batch(self):
        enc = GridEncoder(16, 128)
        x = torch.randn(1, 16, 64, 64)
        assert enc(x).shape == (1, 128)

    def test_large_batch(self):
        enc = GridEncoder(16, 256)
        x = torch.randn(32, 16, 64, 64)
        assert enc(x).shape == (32, 256)


class TestActionHead:
    def test_output_shapes(self):
        head = ActionHead(256, 8, 64)
        emb = torch.randn(4, 256)
        a, x, y = head(emb)
        assert a.shape == (4, 8)
        assert x.shape == (4, 64)
        assert y.shape == (4, 64)

    def test_masking(self):
        head = ActionHead(256, 8, 64)
        emb = torch.randn(2, 256)
        mask = torch.zeros(2, 8, dtype=torch.bool)
        mask[:, 1] = True  # Only ACTION1 available
        a_logits, _, _ = head(emb, mask)
        assert (a_logits[:, 0] == float('-inf')).all()
        assert (a_logits[:, 1] != float('-inf')).all()

    def test_sample_returns_valid(self):
        head = ActionHead(256, 8, 64)
        emb = torch.randn(4, 256)
        mask = torch.ones(4, 8, dtype=torch.bool)
        action, x, y, lp, ent = head.sample(emb, mask)
        assert action.shape == (4,)
        assert (action >= 0).all() and (action < 8).all()
        assert (x >= 0).all() and (x < 64).all()
        assert torch.isfinite(lp).all()
        assert (ent >= 0).all()

    def test_log_prob_finite(self):
        head = ActionHead(256, 8, 64)
        emb = torch.randn(4, 256)
        at = torch.randint(0, 8, (4,))
        x = torch.randint(0, 64, (4,))
        y = torch.randint(0, 64, (4,))
        lp, ent = head.log_prob(emb, at, x, y)
        assert torch.isfinite(lp).all()

    def test_entropy_nonneg(self):
        head = ActionHead(256, 8, 64)
        emb = torch.randn(4, 256)
        _, _, _, _, ent = head.sample(emb)
        assert (ent >= 0).all()


class TestPolicyNetwork:
    def test_forward(self):
        pol = PolicyNetwork(16, 256, 512)
        grid = torch.randn(2, 16, 64, 64)
        aux = torch.randn(2, 15)
        mask = torch.ones(2, 8, dtype=torch.bool)
        a, x, y = pol(grid, aux, mask)
        assert a.shape == (2, 8)

    def test_sample(self):
        pol = PolicyNetwork(16, 256, 512)
        grid = torch.randn(2, 16, 64, 64)
        aux = torch.randn(2, 15)
        action, x, y, lp, ent = pol.sample(grid, aux)
        assert action.shape == (2,)
        assert torch.isfinite(lp).all()

    def test_evaluate(self):
        pol = PolicyNetwork(16, 256, 512)
        grid = torch.randn(2, 16, 64, 64)
        aux = torch.randn(2, 15)
        at = torch.randint(0, 8, (2,))
        x = torch.randint(0, 64, (2,))
        y = torch.randint(0, 64, (2,))
        lp, ent = pol.evaluate(grid, aux, at, x, y)
        assert lp.shape == (2,)

    def test_gradient_flow(self):
        pol = PolicyNetwork(16, 128, 256)
        grid = torch.randn(2, 16, 64, 64)
        aux = torch.randn(2, 15)
        _, _, _, lp, _ = pol.sample(grid, aux)
        loss = -lp.mean()
        loss.backward()
        for p in pol.parameters():
            if p.requires_grad:
                assert p.grad is not None
                break


class TestValueNetwork:
    def test_forward_shape(self):
        val = ValueNetwork(16, 256, 512)
        grid = torch.randn(2, 16, 64, 64)
        aux = torch.randn(2, 15)
        v = val(grid, aux)
        assert v.shape == (2,)

    def test_gradient_flow(self):
        val = ValueNetwork(16, 128, 256)
        grid = torch.randn(2, 16, 64, 64)
        aux = torch.randn(2, 15)
        v = val(grid, aux)
        loss = v.mean()
        loss.backward()
        for p in val.parameters():
            if p.requires_grad:
                assert p.grad is not None
                break
