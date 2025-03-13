from torch.nn import Sequential


class UnpackingSequential(Sequential):
    def forward(self, *args: any) -> any:
        for module in self:
            args = module(*args)
        return args
