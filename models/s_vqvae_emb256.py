import torch
from torch import nn
from torch.nn import functional as F
import distributed as dist_fn

# vqvae quantization
class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()
        self.dim = dim   # dimenionaility of latent space
        self.n_embed = n_embed   # number of codebook vectors
        self.decay = decay   # EMA decay for notebook updates
        self.eps = eps   # avoid zero division

        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input):
        flatten = input.reshape(-1, self.dim)
        
        # calculate distance input vectors to codebook vectors and find nearest
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        
        # onehot encoding for easier update calculations
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        
        # quantization and codebook lookup
        quantize = self.embed_code(embed_ind)

        
        if self.training:
            # cluster statistics
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            # GPU synchronization
            dist_fn.all_reduce(embed_onehot_sum)
            dist_fn.all_reduce(embed_sum)

            # EMA codebook vector update
            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        # commitment loss and gradient flow
        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))

# conv block for improved feature extraction
class ConvBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
            nn.Dropout(0.5)
        )

    def forward(self, input):
        out = self.conv(input)
        out += input
        return out

# encoder
class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_conv_block, n_conv_channel, stride):
        super().__init__()
        blocks = [
            nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(channel // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(channel, channel, 3, padding=1),
        ]

        for _ in range(n_conv_block):
            blocks.append(ConvBlock(channel, n_conv_channel))

        blocks.append(nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)

# decoder
class Decoder(nn.Module):
    def __init__(self, in_channel, out_channel, channel, n_conv_block, n_conv_channel, stride):
        super().__init__()
        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]

        for _ in range(n_conv_block):
            blocks.append(ConvBlock(channel, n_conv_channel))

        blocks.append(nn.ReLU(inplace=True))
        blocks.append(nn.Dropout(0.5))

        if stride == 4:
            blocks.extend([
                nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                nn.BatchNorm2d(channel // 2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(channel // 2, out_channel, 4, stride=2, padding=1),
            ])
        elif stride == 2:
            blocks.append(nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)

# model class
class S_VQVAE(nn.Module):
    def __init__(
        self,
        in_channel=3,
        channel=128,
        n_conv_block=2,
        n_conv_channel=32,
        embed_dim=32,
        n_embed=256,
        num_classes=10,
        decay=0.99,
    ):
        super().__init__()

        # encoder, quantizer, decoder for reconstruction
        self.encoder = Encoder(in_channel, channel, n_conv_block, n_conv_channel, stride=2)
        self.quantize_conv = nn.Conv2d(channel, embed_dim, 1)
        self.quantize = Quantize(embed_dim, n_embed)
        self.decoder = Decoder(embed_dim, in_channel, channel, n_conv_block, n_conv_channel, stride=4)

        # classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(embed_dim, num_classes),
            nn.Dropout(0.5), 
        )

    def forward(self, input):
        # encoding and quantization
        quant, diff, embed_ind = self.encode(input)
        # reconstruction
        dec = self.decode(quant)
        # classification from quantized representation
        logits = self.classify(quant)
        return dec, diff, logits

    def encode(self, input):
        enc = self.encoder(input)
        quant = self.quantize_conv(enc).permute(0, 2, 3, 1)
        quant, diff, embed_ind = self.quantize(quant)
        return quant.permute(0, 3, 1, 2), diff, embed_ind

    def decode(self, quant):
        return self.decoder(quant)

    def classify(self, quant):
        return self.classifier(quant)

    def decode_code(self, code):
        quant = self.quantize.embed_code(code).permute(0, 3, 1, 2)
        return self.decode(quant)

