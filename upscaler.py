import torch

class ModelUpscaler(torch.nn.Module):
    def __init__(self, model, upscale):
        super(ModelUpscaler, self).__init__()
        self.model = model
        self.upscale = upscale
        self.upscaler = torch.nn.Upsample((self.upscale, self.upscale))
        
    def forward(self, x):
        x = self.upscaler(x)
        x = self.model(x)
        return x
