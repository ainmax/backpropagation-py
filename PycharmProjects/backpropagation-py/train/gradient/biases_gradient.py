from dataclasses import dataclass

from train.gradient.network_gradient import NetworkGradient


@dataclass(kw_only=True)
class BiasesGradient(NetworkGradient):
    def _calc_linear_derivative(self, linear_coefficient: float) -> float:
        return 1

    def _get_connected_parameters_count(self, previous_layer_size: int, current_layer_size: int) -> int:
        return current_layer_size
