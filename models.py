import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

NUM_CLASSES = 4
HIDDEN      = 512
DROPOUT     = 0.3
LORA_RANK   = 16
LORA_ALPHA  = 32
LORA_LAYERS = [8, 9, 10, 11]


class LoRALayer(nn.Module):
    def __init__(self, orig: nn.Linear, r: int = LORA_RANK, alpha: int = LORA_ALPHA):
        super().__init__()
        self.orig  = orig
        self.r     = r
        self.scale = alpha / r
        d_in, d_out = orig.in_features, orig.out_features
        self.A = nn.Parameter(torch.randn(d_in, r) * 0.01)
        self.B = nn.Parameter(torch.zeros(r, d_out))
        for p in self.orig.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        return self.orig(x) + (x @ self.A @ self.B) * self.scale


def _get_feat_extract_output_lengths(input_lengths):
    def _conv_out(length, k, s):
        return torch.div(length - k, s, rounding_mode="floor") + 1
    for k, s in zip([10, 3, 3, 3, 3, 2, 2], [5, 2, 2, 2, 2, 2, 2]):
        input_lengths = _conv_out(input_lengths, k, s)
    return input_lengths


def _make_feature_mask(attention_mask):
    input_lengths = attention_mask.sum(-1).long()
    feat_lengths  = _get_feat_extract_output_lengths(input_lengths)
    B      = attention_mask.shape[0]
    T_feat = int(feat_lengths.max().item())
    mask   = torch.zeros(B, T_feat, device=attention_mask.device)
    for i, fl in enumerate(feat_lengths):
        mask[i, :int(fl.item())] = 1.0
    return mask


class FairSERModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base",
            local_files_only=True
        )
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        for layer_idx in LORA_LAYERS:
            layer = self.backbone.encoder.layers[layer_idx]
            layer.attention.q_proj = LoRALayer(layer.attention.q_proj)
            layer.attention.v_proj = LoRALayer(layer.attention.v_proj)
        self.head = nn.Sequential(
            nn.Linear(768, HIDDEN),
            nn.LayerNorm(HIDDEN),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN, HIDDEN // 2),
            nn.LayerNorm(HIDDEN // 2),
            nn.GELU(),
            nn.Dropout(DROPOUT * 0.7),
        )
        self.classifier = nn.Linear(HIDDEN // 2, NUM_CLASSES)

    def unfreeze_transformer_layers(self, layer_indices):
        for idx in layer_indices:
            for p in self.backbone.encoder.layers[idx].parameters():
                p.requires_grad_(True)
        print(f"  [Model] Unfroze transformer layers: {layer_indices}")

    def unfreeze_feature_extractor(self):
        for p in self.backbone.feature_extractor.parameters():
            p.requires_grad_(True)
        for p in self.backbone.feature_projection.parameters():
            p.requires_grad_(True)
        print("  [Model] Unfroze CNN feature extractor + feature projection")

    def unfreeze_all(self):
        for p in self.backbone.parameters():
            p.requires_grad_(True)
        print("  [Model] Unfroze entire backbone")

    def get_param_groups(self, cnn_lr=1e-6, transformer_lr=5e-6, head_lr=1e-4):
        lora_params = []
        lora_ids    = set()
        for layer in self.backbone.encoder.layers:
            for m in [layer.attention.q_proj, layer.attention.v_proj]:
                if isinstance(m, LoRALayer):
                    lora_params += [m.A, m.B]
                    lora_ids.update([id(m.A), id(m.B)])
        cnn_params = [
            p for p in
            list(self.backbone.feature_extractor.parameters()) +
            list(self.backbone.feature_projection.parameters())
            if p.requires_grad and id(p) not in lora_ids
        ]
        trans_params = [
            p for p in self.backbone.encoder.parameters()
            if p.requires_grad and id(p) not in lora_ids
        ]
        head_params = (
            list(self.head.parameters()) +
            list(self.classifier.parameters())
        )
        groups = []
        if cnn_params:
            groups.append({"params": cnn_params,   "lr": cnn_lr,         "name": "cnn"})
        if trans_params:
            groups.append({"params": trans_params,  "lr": transformer_lr, "name": "transformer"})
        if lora_params:
            groups.append({"params": lora_params,   "lr": head_lr,        "name": "lora"})
        groups.append(    {"params": head_params,    "lr": head_lr,        "name": "head"})
        return groups

    def _pool(self, hidden_states, attention_mask=None):
        if attention_mask is not None:
            feat_mask = _make_feature_mask(attention_mask)
            T = hidden_states.size(1)
            if feat_mask.size(1) > T:
                feat_mask = feat_mask[:, :T]
            elif feat_mask.size(1) < T:
                pad = torch.zeros(feat_mask.size(0), T - feat_mask.size(1),
                                  device=feat_mask.device)
                feat_mask = torch.cat([feat_mask, pad], dim=1)
            mask  = feat_mask.unsqueeze(-1)
            denom = mask.sum(1).clamp(min=1e-8)
            return (hidden_states * mask).sum(1) / denom
        return hidden_states.mean(1)

    def forward(self, input_values, attention_mask=None):
        out = self.backbone(input_values, attention_mask=attention_mask)
        return self.classifier(self.head(
            self._pool(out.last_hidden_state, attention_mask)
        ))

    def get_penultimate(self, input_values, attention_mask=None):
        out = self.backbone(input_values, attention_mask=attention_mask)
        return self.head(self._pool(out.last_hidden_state, attention_mask))

    def trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def param_summary(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = self.trainable_params()
        print(f"  Total params     : {total:,}")
        print(f"  Trainable params : {trainable:,}  ({100 * trainable / total:.1f}%)")
