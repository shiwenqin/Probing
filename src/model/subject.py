from abc import abstractmethod, ABC

import torch
from torch.nn import Module
from transformers import AutoModel


class Subject(Module, ABC):

    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.bert = self._build_bert()

    @abstractmethod
    def _build_bert(self):
        """Initialize and return the bert model.

        :return: the bert model.
        """

    def forward(self, inputs):
        outputs = self.bert.forward(**inputs, output_hidden_states=True)
        return outputs.hidden_states


class HuggingFace(Subject):
    """A Pre-Trained BERT from the huggingface transformers library that returns hidden states of all layers, including
    the initial matrix embeddings (layer 0).
    """

    def __init__(self, model_name):
        """

        :param model_name: huggingface model name or path to model checkpoint file
        """
        super().__init__(model_name)

    def _build_bert(self):
        model = AutoModel.from_pretrained(self.model_name)
        return model
