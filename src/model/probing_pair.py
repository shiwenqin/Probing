import torch.nn as nn

class ProbingPair(nn.Module):

    def __init__(self, subject, pooler, probe, freeze_subject=True):
        super().__init__()
        self.subject_model = subject
        self.pooler = pooler
        self.probe = probe

        if freeze_subject:
            self._freeze_model(subject)

    @staticmethod
    def _freeze_model(model):
        """Exclude all parameters of `model` from any gradient updates.

        :param model: freeze all parameters of this model.
        """
        for param in model.parameters():
            param.requires_grad = False

    def get_state_dict(self):
        return self.subject_model.state_dict()

    def forward(self, inputs):
        """
        :param inputs: a tuple of subject model inputs, and target spans. The spans are needed for the pooler to select
        and pool the correct embeddings.
        """

        x, target_spans = inputs
        hidden_states = self.subject_model(x)
        # print("x: ",x['input_ids'].shape)
        # print("hidden_state: ",hidden_states[0].shape)
        # print("target_spans: ",target_spans.shape)
        x = self.pooler(hidden_states, target_spans)
        y = self.probe(x)
        # print("y: ",y.shape)
        return y