from dataclasses import dataclass
from typing import Optional

@dataclass
class JEPAConfig:
    lambda_jepa: float = 1.0        # JEPA loss weight
    gamma_ntp: float = 1.0          # NTP loss weight
    k_pred_tok: int = 1             # Number of PRED tokens
    loss_dropout: float = 0.0       # Dropout percentage of JEPA loss
    distance_metric: str = 'cosine' # Loss alg used for JEPA
    use_jepa: bool = True           # Enable/disable JEPA

    def __post_init__(self):
        assert self.lambda_jepa >= 0, "lambda_jepa must be non-negative"
        assert self.gamma_ntp >= 0, "gamma_ntp must be non-negative"
        assert 0 <= self.k_pred_tok <= 4, "k should be between 0 and 4"
        assert 0 <= self.loss_dropout < 1, "LD must be between 0 and 1"
        assert self.distance_metric in ['cosine', 'l2', 'mse']

@dataclass
class GPT2JEPAConfig:
     
    jepa_config: Optional[JEPAConfig] = None

    def __post_init__(self):
        if self.jepa_config is None:
            self.jepa_config = JEPAConfig()