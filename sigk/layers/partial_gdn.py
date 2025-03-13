from sigk.layers.gdn import GDN, IGDN

from torch import Tensor


class PartialGDN(GDN):
    def forward(self, tensor: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        out_mask = tensor.movedim(1, -1).prod(dim=-1).movedim(-1, 1)
        out_mask = out_mask.repeat(1, mask.shape[1], 1, 1)
        return super().forward(tensor * mask) * out_mask, out_mask


class PartialIGDN(IGDN):
    def forward(self, tensor: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        out_mask = tensor.movedim(1, -1).prod(dim=-1).movedim(-1, 1)
        out_mask = out_mask.repeat(1, mask.shape[1], 1, 1)
        return super().forward(tensor * mask) * out_mask, out_mask
