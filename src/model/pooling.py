import torch
import torch.nn as nn

class AttentionPooler(nn.Module):

    def __init__(self, input_dim, hidden_dim, layer_to_probe, single_span, concat=False):
        super().__init__()

        self.concat = concat
        if concat and layer_to_probe != 0:
            input_dim *= 2

        self.in_projection = nn.ModuleList([nn.Linear(input_dim, hidden_dim),
                                            nn.Linear(input_dim, hidden_dim)])
        self.attention_scorers = nn.ModuleList([nn.Linear(hidden_dim, 1, bias=False),
                                                nn.Linear(hidden_dim, 1, bias=False)])

        self.layer_to_probe = layer_to_probe
        self.single_span = single_span

    def forward(self, hidden_states, target_spans):
        """
        :param hidden_states: the hidden states of the subject model with shape (N, L, D), i.e. batch size,
        sequence length and hidden dimension.
        :param target_spans: a tensor of spans where each span is represented as triple: (i, a, b) with i being the
        span's position in the batch, and [a, b) the span interval along the sequence dimension.
        :return: a tensor of pooled span embeddings of dimension `hidden_dim` or 2 * `hidden_dim` for pair tasks.
        """

        if not self.single_span:
            pooled_span1 = self._single_span_pool(hidden_states, target_spans[0], 0)
            pooled_span2 = self._single_span_pool(hidden_states, target_spans[1], 1)

            return torch.cat([pooled_span1, pooled_span2], dim=1)

        return self._single_span_pool(hidden_states, target_spans, 0)

    def _single_span_pool(self, hidden_states, target_spans, k):
        """
        :param hidden_states: the hidden states of the subject model with shape (N, L, D), i.e. batch size,
        sequence length and hidden dimension.
        :param target_spans: a tensor of spans where each span is represented as triple: (i, a, b) with i being the
        span's position in the batch, and [a, b) the span interval along the sequence dimension.
        :param k: one of two span specific parameter sets to use with `k` in [0, 1].
        :return: a tensor of pooled span embeddings.
        """
        
        if self.concat and self.layer_to_probe != 0:
            hidden_states = torch.cat([hidden_states[self.layer_to_probe], hidden_states[0]], dim=-1)
        else:
            hidden_states = hidden_states[self.layer_to_probe]

        embed_spans = [hidden_states[i, a:b] for i, a, b in target_spans]

        # apply projections with parameters for span target k
        embed_spans = [self.in_projection[k](span) for span in embed_spans]
        att_vectors = [self.attention_scorers[k](span).softmax(0)
                       for span in embed_spans]

        pooled_spans = [att_vec.T @ embed_span
                        for att_vec, embed_span in zip(embed_spans, att_vectors)]

        return torch.stack(pooled_spans).squeeze(-1)
