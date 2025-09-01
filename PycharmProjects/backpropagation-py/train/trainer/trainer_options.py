from dataclasses import dataclass


@dataclass(kw_only=True)
class TrainerOptions:
    learn_speed: float
    inertia_coefficient: float
    epochs_count: int
    batch_size: int
    max_acceptable_average_loss: float
