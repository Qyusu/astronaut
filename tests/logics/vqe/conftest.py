from typing import Any

import pandas as pd
import pytest
from pytest_mock import MockFixture

import astronaut.schema as base_schema
from astronaut.logics.vqe.molecule_configs import (
    BondLengthSettings,
    Structures,
    TargetMolecule,
)


@pytest.fixture
def mock_experiment(mocker: MockFixture) -> MockFixture:
    return mocker.Mock()


@pytest.fixture
def mock_llm_client(mocker: MockFixture) -> MockFixture:
    return mocker.Mock()


@pytest.fixture
def run_context() -> base_schema.RunContext:
    model_versions = base_schema.ModelVersions(
        default="test-default",
        expanding="test-expanding",
        idea="test-idea",
        scoring="test-scoring",
        summary="test-summary",
        expert="test-expert",
        reflection="test-reflection",
        reference="test-ref",
        code="test-code",
        review="test-review",
    )
    return base_schema.RunContext(
        gen_config_dirc="test_configs",
        gen_code_dirc="test_code",
        local_paper_dir="test_papers",
        model_versions=model_versions,
        max_trial_num=3,
        max_idea_num=1,
        max_suggestion_num=5,
        max_scoring_round=1,
        reflection_config=base_schema.ReflectionConfig(
            type=base_schema.ReflectionType.SELF_REFLECTION
        ),
        last_code="print(1)",
        total_cost=0.0,
        need_idea_review=False,
        review_comment="None",
        last_trial_results="None",
        score_histories="None",
        message_history=base_schema.MessageHistory(),
        n_message_history=base_schema.MessageHistoryNum(),
        eval_result_df=pd.DataFrame(),
    )


@pytest.fixture
def molecule_settings() -> TargetMolecule:
    return TargetMolecule(
        name="H2",
        symbols=["H", "H"],
        n_qubits=4,
        structures=Structures(
            charge=0, multi=1, active_electrons=2, active_orbitals=2
        ),
        bond_length_settings=BondLengthSettings(
            min_bond_length=0.5,
            max_bond_length=1.5,
            sample_nums=3,
            equilibrium_bond_length=0.741,
        ),
        exact_energy="fci_energy",
    )


@pytest.fixture
def valid_dataset_configs(molecule_settings: TargetMolecule) -> dict[str, Any]:
    return {"H2": molecule_settings.model_dump()}


@pytest.fixture
def mock_generated_results() -> base_schema.GeneratedResults:
    idea1 = base_schema.GeneratedIdea(
        name="Idea 1",
        summary="S1",
        explanation="E1",
        formula="F1",
        key_sentences=["K1"],
    )
    res1 = base_schema.GeneratedResult(
        idea=idea1,
        score=None,
        implement=base_schema.GeneratedImpl(class_name="Ansatz1", code="code1"),
    )
    return base_schema.GeneratedResults(results=[res1])


@pytest.fixture
def trial_result_df() -> pd.DataFrame:
    import astronaut.logics.vqe.constants as vqe_const

    return pd.DataFrame(
        [
            {
                vqe_const.IDEA_ID_COL: "1_1",
                vqe_const.NUM_GATES_ORDER_COL: "O(N)",
                vqe_const.NUM_PARAMS_ORDER_COL: "O(1)",
                vqe_const.AGGREGATED_NUM_ITER_COL: 100.0,
                vqe_const.AGGREGATED_RUN_TIME_COL: 10.0,
                vqe_const.AGGREGATED_COST_COL: 0.1,
            }
        ]
    )


@pytest.fixture
def raw_result_df() -> pd.DataFrame:
    import astronaut.logics.vqe.constants as vqe_const

    return pd.DataFrame(
        [
            {
                vqe_const.IDEA_ID_COL: "1_1",
                vqe_const.MOLECULE_NAME_COL: "H2",
                vqe_const.BOND_LENGTH_COL: 0.74,
                vqe_const.COST_COL: 0.001,
            }
        ]
    )
