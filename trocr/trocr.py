import torch.nn as nn

class TrOCREmbedder(nn.Module):
    def __init__(super, self):
        super().__init__()

    def forward(self, image):
        return image


class TrOCRUnembedder(nn.Module):
    def __init__(super, self):
        super().__init__()

    def forward(self, logits):
        return np.argmax(logits, axis=1)


class TrOCREncoder(nn.Module):
    def __init__(super, self):
        super().__init__()

    def forward(self, embedding):
        return embedding


class TrOCRDecoder(nn.Module):
    def __init__(super, self):
        super().__init__()

    def forward(self, embedding):
        return embedding


class TrOCR(nn.module):
    def __init__(super, self):
        self.embed = TrOCREmbedder()

        self.encoder = TrOCREncoder()
        self.decoder = TrOCRDecoder()

        self.unembed = TrOCRUnembedder()

    def forward(self, image):
        embedding = self.embed(image)

        encoding = self.encoder(embedding)
        logits = self.decoder(encoding)

        return self.unembed(logits)