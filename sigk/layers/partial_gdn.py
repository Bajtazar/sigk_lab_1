from sigk.layers.gdn import GDN, IGDN

from torch import Tensor


class PartialGDN(GDN):
    def forward(self, tensor: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        return super().forward(tensor * mask) * mask, mask


class PartialIGDN(IGDN):
    def forward(self, tensor: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        return super().forward(tensor * mask) * mask, mask
