# Exact MIL architecture recovered from the original Final159 inference notebook.
# Do not change class/parameter names without regenerating checkpoints.
import torch
from torch import nn

class GatedAttentionHead(
    nn.Module
):
    """
    One independent gated-attention head.

    Input:
        a_in: [n_cells, cell_embed_dim + treatment_embed_dim]

    Output:
        scores: [n_cells]
    """

    def __init__(
        self,
        input_dim,
        attention_dim,
    ):
        super().__init__()

        self.attn_v = nn.Linear(
            input_dim,
            attention_dim,
        )

        self.attn_u = nn.Linear(
            input_dim,
            attention_dim,
        )

        self.attn_w = nn.Linear(
            attention_dim,
            1,
            bias=False,
        )


    def forward(
        self,
        a_in,
    ):
        gated = (
            torch.tanh(
                self.attn_v(
                    a_in
                )
            )
            *
            torch.sigmoid(
                self.attn_u(
                    a_in
                )
            )
        )

        scores = (
            self.attn_w(
                gated
            )
            .squeeze(1)
        )

        return scores


class TreatmentConditionedMultiHeadAttentionMIL(
    nn.Module
):
    def __init__(
        self,
        n_genes,
        n_treatments,
        n_attention_heads=4,
        cell_embed_dim=128,
        treatment_embed_dim=16,
        attention_dim=64,
        classifier_hidden=64,
        cell_dropout=0.20,
        patient_dropout=0.30,
    ):
        super().__init__()

        self.n_genes = int(
            n_genes
        )

        self.n_treatments = int(
            n_treatments
        )

        self.n_attention_heads = int(
            n_attention_heads
        )

        self.cell_embed_dim = int(
            cell_embed_dim
        )

        # Sparse-compatible first projection.
        self.gene_weight = nn.Parameter(
            torch.empty(
                cell_embed_dim,
                n_genes,
            )
        )

        self.gene_bias = nn.Parameter(
            torch.zeros(
                cell_embed_dim
            )
        )

        self.cell_norm = nn.LayerNorm(
            cell_embed_dim
        )

        self.cell_dropout = nn.Dropout(
            cell_dropout
        )

        self.treatment_embedding = (
            nn.Embedding(
                n_treatments,
                treatment_embed_dim,
            )
        )

        attn_input_dim = (
            cell_embed_dim
            + treatment_embed_dim
        )

        # Independent gated-attention modules.
        self.attention_heads = (
            nn.ModuleList(
                [
                    GatedAttentionHead(
                        input_dim=(
                            attn_input_dim
                        ),
                        attention_dim=(
                            attention_dim
                        ),
                    )
                    for _ in range(
                        self.n_attention_heads
                    )
                ]
            )
        )

        # Concatenate all head-specific patient representations.
        classifier_input_dim = (
            self.n_attention_heads
            * cell_embed_dim
            + treatment_embed_dim
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                classifier_input_dim,
                classifier_hidden,
            ),
            nn.ReLU(),
            nn.Dropout(
                patient_dropout
            ),
            nn.Linear(
                classifier_hidden,
                1,
            ),
        )

        self.reset_parameters()


    def reset_parameters(
        self,
    ):
        nn.init.xavier_uniform_(
            self.gene_weight
        )

        nn.init.normal_(
            self.treatment_embedding.weight,
            mean=0.0,
            std=0.05,
        )


    def encode_cells(
        self,
        x_sparse,
    ):
        h = torch.sparse.mm(
            x_sparse,
            self.gene_weight.t(),
        )

        h = h + self.gene_bias
        h = torch.relu(h)
        h = self.cell_norm(h)
        h = self.cell_dropout(h)

        return h


    def get_treatment_embedding(
        self,
        treatment_index,
    ):
        if not torch.is_tensor(
            treatment_index
        ):
            treatment_index = (
                torch.tensor(
                    treatment_index,
                    dtype=torch.long,
                    device=(
                        self.gene_weight.device
                    ),
                )
            )

        treatment_index = (
            treatment_index
            .long()
            .view(1)
        )

        return (
            self.treatment_embedding(
                treatment_index
            )
            .squeeze(0)
        )


    def attention_scores_from_h(
        self,
        h,
        treatment_index,
    ):
        """
        Returns:
            scores: [n_cells, n_heads]
        """

        t = (
            self
            .get_treatment_embedding(
                treatment_index
            )
        )

        t_rep = (
            t.unsqueeze(0)
            .expand(
                h.shape[0],
                -1,
            )
        )

        a_in = torch.cat(
            [
                h,
                t_rep,
            ],
            dim=1,
        )

        head_scores = []

        for head in (
            self.attention_heads
        ):
            head_scores.append(
                head(
                    a_in
                )
            )

        scores = torch.stack(
            head_scores,
            dim=1,
        )

        return scores


    def patient_logit_from_h(
        self,
        h,
        treatment_index,
        return_attention=False,
    ):
        # [n_cells, n_heads]
        scores = (
            self
            .attention_scores_from_h(
                h,
                treatment_index,
            )
        )

        # Each head independently sums to 1 across cells.
        alpha = torch.softmax(
            scores,
            dim=0,
        )

        # [n_heads, n_cells] @ [n_cells, cell_embed_dim]
        # -> [n_heads, cell_embed_dim]
        z_heads = (
            alpha.transpose(
                0,
                1,
            )
            @ h
        )

        z_flat = (
            z_heads
            .reshape(
                -1
            )
        )

        t = (
            self
            .get_treatment_embedding(
                treatment_index
            )
        )

        patient_features = (
            torch.cat(
                [
                    z_flat,
                    t,
                ],
                dim=0,
            )
        )

        logit = (
            self.classifier(
                patient_features
                .unsqueeze(0)
            )
            .squeeze()
        )

        if return_attention:
            return (
                logit,
                alpha,
                z_heads,
            )

        return logit


    def forward_bag(
        self,
        x_sparse,
        treatment_index,
        return_attention=False,
    ):
        h = self.encode_cells(
            x_sparse
        )

        return (
            self
            .patient_logit_from_h(
                h,
                treatment_index,
                return_attention=(
                    return_attention
                ),
            )
        )