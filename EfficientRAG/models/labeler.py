"""
Labeler model: DebertaForSequenceTokenClassification.

Dual-headed DeBERTa model that performs:
  1. Token-level classification (useful/useless per chunk token)
  2. Sequence-level classification (CONTINUE/TERMINATE/FINISH per chunk)
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import DebertaV2Config, DebertaV2Model, DebertaV2PreTrainedModel
from transformers.utils import ModelOutput


@dataclass
class SequenceTokenClassifierOutput(ModelOutput):
    """Output of DebertaForSequenceTokenClassification."""

    loss: Optional[torch.FloatTensor] = None
    sequence_logits: Optional[torch.FloatTensor] = None
    token_logits: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None


class ContextPooler(nn.Module):
    """Pools the [CLS] token hidden state for sequence classification."""

    def __init__(self, config: DebertaV2Config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.pooler_dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Take [CLS] token (first token)
        cls_token = hidden_states[:, 0]
        pooled = self.dropout(cls_token)
        pooled = self.dense(pooled)
        pooled = torch.tanh(pooled)
        return pooled


class DebertaForSequenceTokenClassification(DebertaV2PreTrainedModel):
    """
    DeBERTa-v3 model with two classification heads:
      - Token classification: binary per-token labels (useful=1 / useless=0)
      - Sequence classification: chunk-level tag (CONTINUE / TERMINATE / FINISH)

    This is the Labeler component of EfficientRAG.
    """

    def __init__(self, config: DebertaV2Config):
        super().__init__(config)

        self.num_token_labels = 2  # useful / useless
        self.num_sequence_labels = getattr(config, "num_sequence_labels", 2)

        self.deberta = DebertaV2Model(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Token classification head
        self.token_classifier = nn.Linear(config.hidden_size, self.num_token_labels)

        # Sequence classification head
        self.pooler = ContextPooler(config)
        self.sequence_classifier = nn.Linear(config.hidden_size, self.num_sequence_labels)

        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        token_labels: Optional[torch.Tensor] = None,
        sequence_labels: Optional[torch.Tensor] = None,
        token_label_mask: Optional[torch.Tensor] = None,
        token_pos_weight: Optional[float] = None,
        token_neg_weight: Optional[float] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> SequenceTokenClassifierOutput:
        """
        Args:
            input_ids: [batch, seq_len] - tokenized input
            attention_mask: [batch, seq_len] - attention mask
            token_labels: [batch, seq_len] - per-token binary labels (0/1)
            sequence_labels: [batch] - sequence-level label (0=CONTINUE, 1=TERMINATE, ...)
            token_label_mask: [batch, seq_len] - True for tokens that have meaningful labels
                (False for [CLS], [SEP], [PAD], query tokens)
            token_pos_weight: weight for positive class in token loss
            token_neg_weight: weight for negative class in token loss
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )

        hidden_states = outputs[0]  # [batch, seq_len, hidden]
        hidden_states = self.dropout(hidden_states)

        # Token classification
        token_logits = self.token_classifier(hidden_states)  # [batch, seq_len, 2]

        # Sequence classification
        pooled = self.pooler(hidden_states)
        sequence_logits = self.sequence_classifier(pooled)  # [batch, num_seq_labels]

        loss = None
        if token_labels is not None and sequence_labels is not None:
            loss = self._compute_loss(
                token_logits=token_logits,
                sequence_logits=sequence_logits,
                token_labels=token_labels,
                sequence_labels=sequence_labels,
                token_label_mask=token_label_mask,
                token_pos_weight=token_pos_weight,
                token_neg_weight=token_neg_weight,
            )

        return SequenceTokenClassifierOutput(
            loss=loss,
            sequence_logits=sequence_logits,
            token_logits=token_logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions if output_attentions else None,
        )

    def _compute_loss(
        self,
        token_logits: torch.Tensor,
        sequence_logits: torch.Tensor,
        token_labels: torch.Tensor,
        sequence_labels: torch.Tensor,
        token_label_mask: Optional[torch.Tensor] = None,
        token_pos_weight: Optional[float] = None,
        token_neg_weight: Optional[float] = None,
    ) -> torch.Tensor:
        """Combined loss: token CE + sequence CE."""

        # --- Sequence loss ---
        seq_loss_fn = nn.CrossEntropyLoss()
        seq_loss = seq_loss_fn(sequence_logits, sequence_labels)

        # --- Token loss ---
        # Only compute token loss for non-TERMINATE chunks
        # (TERMINATE chunks don't have meaningful token labels)
        from EfficientRAG.config import TAG_MAPPING_TWO, TAG_MAPPING_THREE

        terminate_id_2 = TAG_MAPPING_TWO[
            "<TERMINATE>"
        ]
        terminate_id_3 = TAG_MAPPING_THREE["<TERMINATE>"]

        # Mask out TERMINATE samples
        non_terminate_mask = (sequence_labels != terminate_id_2) & (
            sequence_labels != terminate_id_3
        )

        if non_terminate_mask.any():
            active_token_logits = token_logits[non_terminate_mask]  # [n, seq, 2]
            active_token_labels = token_labels[non_terminate_mask]  # [n, seq]

            if token_label_mask is not None:
                active_mask = token_label_mask[non_terminate_mask]  # [n, seq]
            else:
                active_mask = active_token_labels >= 0

            # Flatten and select active positions
            flat_logits = active_token_logits.view(-1, 2)  # [n*seq, 2]
            flat_labels = active_token_labels.view(-1)  # [n*seq]
            flat_mask = active_mask.view(-1)  # [n*seq]

            active_logits = flat_logits[flat_mask]
            active_labels = flat_labels[flat_mask]

            if active_logits.numel() > 0:
                # Class-weighted cross-entropy
                if token_pos_weight is not None and token_neg_weight is not None:
                    weight = torch.tensor(
                        [token_neg_weight, token_pos_weight],
                        device=active_logits.device,
                        dtype=active_logits.dtype,
                    )
                    token_loss_fn = nn.CrossEntropyLoss(weight=weight)
                else:
                    token_loss_fn = nn.CrossEntropyLoss()
                token_loss = token_loss_fn(active_logits, active_labels)
            else:
                token_loss = torch.tensor(0.0, device=seq_loss.device)
        else:
            token_loss = torch.tensor(0.0, device=seq_loss.device)

        return seq_loss + token_loss
