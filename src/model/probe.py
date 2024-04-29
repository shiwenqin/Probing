import torch.nn as nn

def choose_probe(cfg, task_cfg=None):
    if task_cfg:
        output_dim = task_cfg.num_classes
        single_span = task_cfg.single_span
    else:
        output_dim = cfg.probe.output_dim
        single_span = cfg.probe.single_span
    if cfg.probe.name == 'mlp':
        return MLP(cfg.probe.input_dim, cfg.probe.hidden_dim, output_dim, cfg.probe.dropout, single_span)
    elif cfg.probe.name == 'linear':
        return LinearClassifier(cfg.probe.input_dim, output_dim, single_span)
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