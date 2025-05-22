from typing import Any

from astronaut.logics.base import QuantumStrategy, StrategyType


class StrategyFactory:
    @classmethod
    def create_strategy(cls, strategy_type: StrategyType, **kwargs: Any) -> QuantumStrategy:
        if strategy_type == StrategyType.QUANTUM_KERNEL:
            from astronaut.logics.quantum_kernel import QuantumKernelStrategy

            return QuantumKernelStrategy(**kwargs)
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")


class QuantumAlgorithmContext:
    def __init__(self, strategy_type: StrategyType, **strategy_kwargs: Any) -> None:
        self._strategy = StrategyFactory.create_strategy(strategy_type, **strategy_kwargs)

    def set_strategy(self, strategy_type: StrategyType, **strategy_kwargs: Any) -> None:
        self._strategy = StrategyFactory.create_strategy(strategy_type, **strategy_kwargs)

    def execute_baseline(self, *args: Any, **kwargs: Any) -> None:
        self._strategy.execute_baseline(*args, **kwargs)

    def execute_experiment(self, *args: Any, **kwargs: Any) -> None:
        if self._strategy.validate():
            self._strategy.execute_experiment(*args, **kwargs)
