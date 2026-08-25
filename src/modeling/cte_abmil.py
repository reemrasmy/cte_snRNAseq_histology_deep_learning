import torch
import torch.nn as nn
import torch.nn.functional as F


class ABMILBackbone(nn.Module):
    """
    Shared ABMIL feature + attention module.

    Input:
        tile_embeddings: [N_tiles, 1536]

    Output:
        bag_embedding: [hidden_dim]
        attention_weights: [N_tiles]

    The bag can represent:
        - one slide: all tile embeddings from one WSI
        - one donor: all tile embeddings from all slides/stains for one donor
    """

    def __init__(self, input_dim=1536, hidden_dim=256, dropout=0.25):
        super().__init__()

        self.tile_feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.attention_signal = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

        self.attention_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

        self.attention_score_layer = nn.Linear(hidden_dim, 1)

    def forward(self, tile_embeddings):
        if tile_embeddings.dim() != 2:
            raise ValueError(
                f"Expected [N_tiles, input_dim], got {tile_embeddings.shape}"
            )

        # [N_tiles, 1536] -> [N_tiles, hidden_dim]
        tile_features = self.tile_feature_extractor(tile_embeddings)

        # Gated attention
        attention_signal = self.attention_signal(tile_features)
        attention_gate = self.attention_gate(tile_features)
        gated_attention = attention_signal * attention_gate

        # [N_tiles, hidden_dim] -> [N_tiles, 1]
        raw_attention_scores = self.attention_score_layer(gated_attention)

        # Normalize across tiles so weights sum to 1
        attention_weights = F.softmax(raw_attention_scores, dim=0)

        # Weighted pooling: [N_tiles, hidden_dim] -> [hidden_dim]
        bag_embedding = torch.sum(attention_weights * tile_features, dim=0)

        return bag_embedding, attention_weights.squeeze(1)


class ABMILRegressor(nn.Module):
    """
    ABMIL for continuous targets.

    Example targets:
        - microglia proportion
        - astrocyte proportion
        - complement pathway score
    """

    def __init__(self, input_dim=1536, hidden_dim=256, mlp_hidden_dim=128, output_dim=2, dropout=0.25, head_type="mlp"):
        super().__init__()

        self.backbone = ABMILBackbone(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

        if head_type == "linear":
            self.regression_head = nn.Linear(
                hidden_dim,
                output_dim,
            )

        elif head_type == "mlp":
            self.regression_head = nn.Sequential(
                nn.Linear(hidden_dim, mlp_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden_dim, output_dim),
            )

        else:
            raise ValueError(
                "head_type must be either 'linear' or 'mlp' "
            ) 


    def forward(self, tile_embeddings):
        bag_embedding, attention_weights = self.backbone(tile_embeddings)
        prediction = self.regression_head(bag_embedding)

        return prediction, attention_weights


class ABMILClassifier(nn.Module):
    """
    ABMIL for stage classification.

    Output:
        logits: [num_classes]

    Use with:
        nn.CrossEntropyLoss()
    """

    def __init__(
        self,
        input_dim=1536,
        hidden_dim=256,
        mlp_hidden_dim=128,
        num_classes=2,
        dropout=0.25,
    ):
        super().__init__()

        self.backbone = ABMILBackbone(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        # Bag-level MLP classification head
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, num_classes),
        )

    def forward(self, tile_embeddings):
        bag_embedding, attention_weights = self.backbone(
            tile_embeddings
        )

        logits = self.classification_head(
            bag_embedding
        )

        return logits, attention_weights

class ABMILAgeRegressor(nn.Module):
    """
    ABMIL regression model using image embeddings plus donor age.
    """

    def __init__(
        self,
        input_dim=1536,
        hidden_dim=256,
        mlp_hidden_dim=128,
        output_dim=2,
        dropout=0.25,
    ):
        super().__init__()

        self.backbone = ABMILBackbone(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim + 1, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, output_dim),
        )

    def forward(
        self,
        tile_embeddings: torch.Tensor,
        age: torch.Tensor,
    ):
        bag_embedding, attention_weights = self.backbone(
            tile_embeddings
        )

        # Ensure age has shape [1]
        age = age.reshape(1).to(
            device=bag_embedding.device,
            dtype=bag_embedding.dtype,
        )

        combined_features = torch.cat(
            [bag_embedding, age],
            dim=0,
        )

        prediction = self.regression_head(
            combined_features
        )

        return prediction, attention_weights