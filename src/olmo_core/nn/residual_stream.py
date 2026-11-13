from typing import Optional

import torch
import torch.nn as nn


class ResidualStream(nn.Module):
    """
    A parameter-free module that just handles a residual stream connection, like those in a transformer
    block. The benefit of using this module instead of a direct add operation is that the flexible
    to configure hooks for logging or other purposes, like with the
    :class:`olmo_core.train.callbacks.GAPMonitorCallback`.

    :param alpha: Scale applied to the branch output before the residual add.
    :param dropout: Dropout probability applied uniformly to every token.
    :param masked_dropout: Dropout probability applied *only* to tokens selected by the
        ``drop_mask`` passed to :meth:`forward` (mm_olmo's ``Dropout(mask_p=...)``, used for
        Molmo2's ``response_residual_dropout``). Tokens outside the mask keep the
        :data:`dropout` rate. Defaults to ``0.0``, in which case behaviour is unchanged.
    """

    def __init__(self, alpha: float = 1.0, dropout: float = 0.0, masked_dropout: float = 0.0):
        super().__init__()
        if not 0.0 <= masked_dropout < 1.0:
            raise ValueError(f"'masked_dropout' must be in [0, 1), got {masked_dropout}")
        self.alpha = alpha
        self.p = dropout
        self.masked_dropout = masked_dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def _masked_dropout(self, x: torch.Tensor, drop_mask: torch.Tensor) -> torch.Tensor:
        """Inverted dropout whose *rate* varies per token, but which samples per **element**.

        Mirrors mm_olmo's ``Dropout.forward``: masked tokens keep each activation with
        probability ``1 - masked_dropout``, the rest with ``1 - dropout``.

        The per-element sampling matters. Broadcasting the ``(batch, seq)`` rate to
        ``(batch, seq, 1)`` and drawing one Bernoulli per *token* would zero a token's entire
        residual contribution at once; across a 36-layer model's 72 residual adds that is
        destructive rather than regularising, and it makes training diverge.
        """
        mask = drop_mask.to(x.dtype)
        keep_prob = mask * (1.0 - self.masked_dropout) + (1.0 - mask) * (1.0 - self.p)
        keep_prob = keep_prob.unsqueeze(-1).broadcast_to(x.shape)
        # Inverted dropout: scale survivors by 1/keep_prob so the expectation is preserved.
        multiplier = torch.empty_like(x).bernoulli_(keep_prob)
        return x * (multiplier / keep_prob)

    def forward(
        self,
        residual: torch.Tensor,
        x: torch.Tensor,
        drop_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        :param residual: The residual stream, ``(batch_size, seq_len, d_model)``.
        :param x: The branch output to add in.
        :param drop_mask: Optional ``(batch_size, seq_len)`` 0/1 tensor selecting the tokens
            that get :data:`masked_dropout`. Required when ``masked_dropout > 0`` and training.
        """
        if self.masked_dropout > 0.0 and self.training:
            if drop_mask is None:
                raise ValueError("'drop_mask' is required when 'masked_dropout' > 0")
            return torch.add(residual, self._masked_dropout(x, drop_mask), alpha=self.alpha)
        return torch.add(residual, self.dropout(x), alpha=self.alpha)
