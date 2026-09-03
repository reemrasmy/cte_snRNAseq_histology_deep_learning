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

    # Input is a UNI2 1536-dimensional embedding vector representing one tile from a given image. 
    # hidden_dim represents the smaller vector UNI2 is converted to that hte network learns
    # super().__init__() initializes the nn.Module that stores the layers/weights of the network 
    def __init__(self, input_dim=1536, hidden_dim=256, dropout=0.25):
        super().__init__()

        self.tile_feature_extractor = nn.Sequential(
            # The layer learning how to convert the UNI2 features into 256 new features 
            nn.Linear(input_dim, hidden_dim), # [1536 --> 256]
            # Changes negative values to 0
            nn.ReLU(),
            # Temporarily masks random intermediate nn output features to prevent overfitting for certain features
            nn.Dropout(dropout),
            
            # Second layer: now creating a 256 vector representation of the tile for the specific prediciton task
            nn.Linear(hidden_dim, hidden_dim), # [256 --> 256]
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # First branch of gated attention mechanism 
        self.attention_signal = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            # Tanh produces a signed attention representation for each tile [-1, 1]
            nn.Tanh()
        )

        # Second branch of gated attention mechanism 
        self.attention_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            # Sigmoid produced attention values between 0 and 1 
            nn.Sigmoid()
        )

        # Collapsing each tile's gated attention representation to one scalar score determining its relative contribution
        self.attention_score_layer = nn.Linear(hidden_dim, 1)

    def forward(self, tile_embeddings):
        # Checking that each bag contains one row per tile and one column per embedding feature [N_tiles, 1536]
        if tile_embeddings.dim() != 2:
            raise ValueError(
                f"Expected [N_tiles, input_dim], got {tile_embeddings.shape}"
            )

        # Converting pretrained UNI2 embeddings into the learned task-specific tile features 
        # [N_tiles, 1536] -> [N_tiles, hidden_dim]
        tile_features = self.tile_feature_extractor(tile_embeddings)

        # Computing the two branches of gated attention (tanh and sigmoid) learned tile features of the same tile
        attention_signal = self.attention_signal(tile_features)
        attention_gate = self.attention_gate(tile_features)

        # Using element-wise multiplication 
        gated_attention = attention_signal * attention_gate

        # Converting each tile's calculated gated representation into a raw attention score [N_tiles, hidden_dim] -> [N_tiles, 1]
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

        # Using the shared feature extractions and attention-pooling strategy
        self.backbone = ABMILBackbone(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

        # Linear head maps the bag representation the input continuous predictive targets
        if head_type == "linear":
            self.regression_head = nn.Linear(
                hidden_dim,
                output_dim,
            )

        # The MLP (added after review with post-doc) -- adds a nonlinear intermediate layer between two linear layers
        # Tested -- improved performance of both classification and regression models
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
        # Pooling all the tile embeddings of the donor's image into one attention-weighted representation 
        bag_embedding, attention_weights = self.backbone(tile_embeddings)
       
       # Predict the continuous target(s) from the bag_embedding. No activation is used here so the regression outputs can take
       # unrestricted negative or positive values
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

        # Using the saame tile feature extractor and attention pooling as regression model
        self.backbone = ABMILBackbone(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        # Mapping the pooled bag to one logit per class (num_classes)
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, num_classes),
        )

    def forward(self, tile_embeddings):
        
        # Converting all tiles from the bag into one learned attention-weighted representation (from ABMILBackbone)
        bag_embedding, attention_weights = self.backbone(
            tile_embeddings
        )

        # Produce a raw score for each possible class (Low and High for this applicaition)
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

        # Making this age informed, by adding one scalar feature to the image embedding representation
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