from sigk.layers.gdn import GDN, IGDN

from torch import Tensor


class PartialGDNBase:
    def _calculate_mask(self, mask: Tensor) -> Tensor:
        out_mask = mask.sum(dim=1).clamp(min=0, max=1).unsqueeze(1)
        return out_mask.repeat(1, mask.shape[1], 1, 1)


class PartialGDN(GDN, PartialGDNBase):
    def forward(self, tensor: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        out_mask = self._calculate_mask(mask)
        return super().forward(tensor * mask) * out_mask, out_mask


class PartialIGDN(IGDN, PartialGDNBase):
    def forward(self, tensor: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        out_mask = self._calculate_mask(mask)
        return super().forward(tensor * mask) * out_mask, out_mask
