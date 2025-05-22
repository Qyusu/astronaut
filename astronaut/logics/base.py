from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any


class StrategyType(Enum):
    QUANTUM_KERNEL = auto()


class QuantumStrategy(ABC):
    @abstractmethod
    def execute_baseline(self, *args: Any, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def execute_experiment(self, *args: Any, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def validate(self) -> bool:
        pass
