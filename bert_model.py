from transformers.models.bert.modeling_bert import *
from torch.nn import MarginRankingLoss
from typing import List, Optional, Tuple, Union

class BertForRanking(BertPreTrainedModel):
    def __init__(self, config, margin_value):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.margin = margin_value

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        stronger_input_ids: Optional[torch.Tensor] = None,
        stronger_attention_mask: Optional[torch.Tensor] = None,
        weaker_input_ids: Optional[torch.Tensor] = None,
        weaker_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        stronger_outputs = self.bert(
            stronger_input_ids,
            attention_mask=stronger_attention_mask,
        )
        stronger_pooled_output = stronger_outputs[1]
        stronger_pooled_output = self.dropout(stronger_pooled_output)
        stronger_logits = self.classifier(stronger_pooled_output).squeeze(dim=-1)

        loss = None
        weaker_logits = None
        if labels is not None:
            assert weaker_input_ids is not None, "weaker output is necessary during training"
            weaker_outputs = self.bert(
                weaker_input_ids,
                attention_mask=weaker_attention_mask,
            )
            weaker_pooled_output = weaker_outputs[1]
            weaker_pooled_output = self.dropout(weaker_pooled_output)
            weaker_logits = self.classifier(weaker_pooled_output).squeeze(dim=-1)

            loss_fct = MarginRankingLoss(margin=self.margin)
            loss = loss_fct(stronger_logits, weaker_logits, labels)
                
        return SequenceClassifierOutput(
            loss=loss,
            logits=stronger_logits,
            hidden_states=stronger_outputs.hidden_states,
            attentions=stronger_outputs.attentions,
        )