from typing import Any

import pytest
from pytest_mock import MockerFixture

from astronaut.logics.base import QuantumStrategy, StrategyType
from astronaut.logics.strategies import QuantumAlgorithmContext, StrategyFactory


class TestStrategyFactory:
    def test_create_strategy_quantum_kernel(
        self, mocker: MockerFixture
    ) -> None:
        mock_qk_strategy_cls = mocker.patch(
            "astronaut.logics.quantum_kernel.QuantumKernelStrategy"
        )
        mock_instance = mock_qk_strategy_cls.return_value

        strategy = StrategyFactory.create_strategy(
            StrategyType.QUANTUM_KERNEL, some_arg="value"
        )

        mock_qk_strategy_cls.assert_called_once_with(some_arg="value")
        assert strategy == mock_instance

    def test_create_strategy_vqe(self, mocker: MockerFixture) -> None:
        mock_vqe_strategy_cls = mocker.patch("astronaut.logics.vqe.VQEStrategy")
        mock_instance = mock_vqe_strategy_cls.return_value

        strategy = StrategyFactory.create_strategy(
            StrategyType.VQE, other_arg=123
        )

        mock_vqe_strategy_cls.assert_called_once_with(other_arg=123)
        assert strategy == mock_instance

    def test_create_strategy_unknown_raises_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown strategy type"):
            StrategyFactory.create_strategy("INVALID_TYPE")  # type: ignore


class TestQuantumAlgorithmContext:
    @pytest.fixture
    def mock_strategy(self, mocker: MockerFixture) -> QuantumStrategy:
        return mocker.Mock(spec=QuantumStrategy)

    @pytest.fixture
    def context(
        self, mocker: MockerFixture, mock_strategy: QuantumStrategy
    ) -> QuantumAlgorithmContext:
        mocker.patch.object(
            StrategyFactory, "create_strategy", return_value=mock_strategy
        )
        return QuantumAlgorithmContext(StrategyType.QUANTUM_KERNEL, arg1="test")

    def test_init(
        self, context: QuantumAlgorithmContext, mock_strategy: QuantumStrategy
    ) -> None:
        assert context._strategy == mock_strategy
        StrategyFactory.create_strategy.assert_called_with(
            StrategyType.QUANTUM_KERNEL, arg1="test"
        )

    def test_set_strategy(
        self,
        context: QuantumAlgorithmContext,
        mocker: MockerFixture,
        mock_strategy: QuantumStrategy,
    ) -> None:
        new_mock_strategy = mocker.Mock(spec=QuantumStrategy)
        mocker.patch.object(
            StrategyFactory, "create_strategy", return_value=new_mock_strategy
        )

        context.set_strategy(StrategyType.VQE, arg2="new")

        assert context._strategy == new_mock_strategy
        StrategyFactory.create_strategy.assert_called_with(
            StrategyType.VQE, arg2="new"
        )

    def test_execute_baseline(
        self, context: QuantumAlgorithmContext, mock_strategy: Any
    ) -> None:
        context.execute_baseline("arg", kw="arg")
        mock_strategy.execute_baseline.assert_called_once_with("arg", kw="arg")

    def test_execute_experiment_valid(
        self, context: QuantumAlgorithmContext, mock_strategy: Any
    ) -> None:
        mock_strategy.validate.return_value = True

        context.execute_experiment("exp_arg", kw="exp")

        mock_strategy.validate.assert_called_once()
        mock_strategy.execute_experiment.assert_called_once_with(
            "exp_arg", kw="exp"
        )

    def test_execute_experiment_invalid(
        self, context: QuantumAlgorithmContext, mock_strategy: Any
    ) -> None:
        mock_strategy.validate.return_value = False

        context.execute_experiment("exp_arg", kw="exp")

        mock_strategy.validate.assert_called_once()
        mock_strategy.execute_experiment.assert_not_called()
