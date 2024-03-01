import torch.nn as nn

def choose_probe(cfg):
    if cfg.probe.name == 'mlp':
        return MLP(cfg.probe.input_dim, cfg.probe.hidden_dim, cfg.probe.output_dim, cfg.probe.dropout, cfg.probe.single_span)
    elif cfg.probe.name == 'linear':
        return LinearClassifier(cfg.probe.input_dim, cfg.probe.output_dim, cfg.probe.single_span)
    else:
        raise ValueError(f'Invalid probe name: {cfg.probe.name}')

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout, single_span):
        super().__init__()

        if not single_span:
            input_dim *= 2

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout

        self.classifier = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.Tanh(),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.output_dim)
        )

    def forward(self, inputs):
        return self.classifier(inputs)
    
class LinearClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, single_span):
        super().__init__()
        
        if not single_span:
            input_dim *= 2

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.classifier = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, inputs):
        return self.classifier(inputs)