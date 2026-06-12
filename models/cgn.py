import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.loss import LabelSmoothingCrossEntropy

# Dataset-specific config: (num_classes, encoder_dim)
DATASET_CONFIG = {
    "humanact12_22": (34, 254),
    "nturgbvibe":    (13, 253),
}

# Zero-padding dims appended to action embedding per dataset
PADDING_DIM = {
    "nturgbd60_sub_vibe": 3,
    "nturgbvibe":         3,
    "humanact12_22":      2,
}


class CGN(nn.Module):
    def __init__(self, njoints, nfeats, num_actions,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4,
                 dropout=0.1, activation="gelu", dataset='amass', dropout_inf=0.0,
                 **kwargs):
        super().__init__()

        self.njoints      = njoints
        self.nfeats       = nfeats
        self.input_feats  = njoints * nfeats
        self.num_actions  = num_actions
        self.dataset      = dataset
        self.latent_dim   = latent_dim
        self.ff_size      = ff_size
        self.num_layers   = num_layers
        self.num_heads    = num_heads
        self.dropout      = dropout
        self.dropout_inf  = dropout_inf
        self.activation   = activation
        self.cond_mask_prob = 0.

        num_classes, encoder_dim = DATASET_CONFIG[dataset]

        # --- Timestep & positional encoding ---
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.embed_timestep       = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        # --- Input projections ---
        self.input_process        = InputProcess(self.input_feats, self.latent_dim)
        self.action_input_process = InputProcess(self.input_feats, encoder_dim - num_classes)

        # --- Semantic encoder ---
        enc_layer = nn.TransformerEncoderLayer(
            d_model         = encoder_dim - num_classes,
            nhead           = self.num_heads,
            dim_feedforward = self.ff_size,
            dropout         = self.dropout,
            activation      = self.activation,
        )
        self.semantic_encoder = nn.TransformerEncoder(enc_layer, num_layers=self.num_layers)

        # --- Decoder ---
        dec_layer = nn.TransformerDecoderLayer(
            d_model         = self.latent_dim,
            nhead           = self.num_heads,
            dim_feedforward = self.ff_size,
            dropout         = self.dropout,
            activation      = self.activation,
        )
        self.semantic_decoder = nn.TransformerDecoder(dec_layer, num_layers=4)

        # --- Output projection ---
        self.output_process = OutputProcess(self.input_feats, self.latent_dim,
                                            self.njoints, self.nfeats)

        # --- Action classification head ---
        dim_feat = 263 if dataset in ("humanact12_22", "nturgbvibe") else 150
        self.action_head = ActionHead(dim_feat=dim_feat, num_classes=num_classes)

        self.criterion = LabelSmoothingCrossEntropy(smoothing=0.1)

        print("CGN init complete")

    # ---------------------------------------------------------------- #

    def _encode_action(self, x_start):
        """Encode x_start through the semantic encoder; return mean-pooled features."""
        action_emb = self.action_input_process(x_start)                  # [T, B, d]
        action_emb = self.semantic_encoder(action_emb.permute(1, 0, 2))  # [B, T, d]
        return action_emb.mean(1)                                         # [B, d]

    def forward(self, x, timesteps, y=None, x_start=None, label=None):
        """
        Args:
            x:          [B, njoints, nfeats, T]  noisy input at timestep t
            timesteps:  [B]                       diffusion timesteps
            y:          optional dict with 'source_motion' and 'label'
            x_start:    reference motion for conditioning
            label:      one-hot action label [B, num_classes]
        """
        bs = x.shape[0]

        if y is not None:
            x_start = y["source_motion"]
            label   = y["label"]

        # Timestep embedding
        emb = self.embed_timestep(timesteps)  # [1, B, latent_dim]

        # Action conditioning
        action_emb = self._encode_action(x_start)                        # [B, d_enc]
        mask = torch.bernoulli(
            torch.ones(bs) * self.cond_mask_prob
        ).view(bs, 1).cuda()  # 1 → drop condition, 0 → use condition

        pad_dim    = PADDING_DIM.get(self.dataset, 0)
        action_emb = torch.cat([action_emb, label.cuda()], dim=1)
        if pad_dim > 0:
            action_emb = torch.cat(
                [action_emb, torch.zeros(bs, pad_dim).cuda()], dim=1
            )

        emb = emb + (action_emb * (1.0 - mask))

        # Transformer decode
        x    = self.input_process(x)
        xseq = torch.cat((emb, x), dim=0)
        xseq = self.sequence_pos_encoder(xseq)
        if self.dropout_inf > 0:
            xseq = F.dropout(xseq, p=self.dropout_inf, training=True)

        xseq = xseq.float()
        emb  = emb.float()

        output = self.semantic_decoder(xseq, emb)[1:]
        output = self.output_process(output)

        # Classification loss
        logit = self.action_head(output.squeeze(2))
        loss  = self.criterion(logit, torch.argmax(label, dim=1).cuda())

        return output, loss


# -------------------------------------------------------------------- #
# Sub-modules                                                           #
# -------------------------------------------------------------------- #

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.sequence_pos_encoder = sequence_pos_encoder
        self.time_embed = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(
            self.sequence_pos_encoder.pe[timesteps]
        ).permute(1, 0, 2)


class InputProcess(nn.Module):
    """Project raw pose features to latent_dim."""
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.pose_embedding = nn.Linear(input_feats, latent_dim)

    def forward(self, x):
        bs, njoints, nfeats, nframes = x.shape
        x = x.permute(3, 0, 1, 2).reshape(nframes, bs, njoints * nfeats)
        return self.pose_embedding(x)  # [T, B, latent_dim]


class OutputProcess(nn.Module):
    """Project latent back to pose space."""
    def __init__(self, input_feats, latent_dim, njoints, nfeats):
        super().__init__()
        self.njoints    = njoints
        self.nfeats     = nfeats
        self.pose_final = nn.Linear(latent_dim, input_feats)

    def forward(self, output):
        nframes, bs, _ = output.shape
        output = self.pose_final(output)                                   # [T, B, input_feats]
        output = output.reshape(nframes, bs, self.njoints, self.nfeats)
        return output.permute(1, 2, 3, 0)                                  # [B, njoints, nfeats, T]


class ActionHead(nn.Module):
    """Two-layer MLP for action classification."""
    def __init__(self, dim_feat=512, num_classes=60):
        super().__init__()
        self.fc  = nn.Linear(dim_feat, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, feat):
        feat = feat.mean(dim=[2])
        return self.fc2(self.fc(feat))