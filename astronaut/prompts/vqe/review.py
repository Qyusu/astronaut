# flake8: noqa

from textwrap import dedent

from astronaut.prompts.vqe.constraints import (
    ANSATZ_CONSTRAINTS,
    COST_FUNCTION_DEFINITION,
    OPTIMIZER_SETTINGS,
)


class ReviewIdeaPrompt:
    def __init__(
        self,
        ansatz_constraints: str = ANSATZ_CONSTRAINTS.strip(),
        cost_function_definition: str = COST_FUNCTION_DEFINITION.strip(),
        optimizer_settings: str = OPTIMIZER_SETTINGS.strip(),
        target_molecule_names: str = "",
        max_suggestion_num: int = 3,
    ) -> None:
        self.ansatz_constraints = ansatz_constraints
        self.cost_function_definition = cost_function_definition
        self.optimizer_settings = optimizer_settings
        self.target_molecule_names = target_molecule_names
        self.max_suggestion_num = max_suggestion_num

        self.developer_template = dedent(
            """
            # Role
            You are an expert in quantum physics and quantum chemistry computation, specializing in variational quantum eigensolver (VQE) ansatz design.


            # Task and Objective
            Your task is to review past VQE ansatz ideas and their performance in ground-state energy estimation. After reviewing the ideas, propose improvements to minimize the cost function and convergence through continuous refinement. Restrict your review and proposal to the ansatz circuit design itself; do NOT propose changes to the Hamiltonian formulation, optimization routine, measurement strategy, or other workflows.

            Encourage innovative approaches that go beyond simple combinations of established methods when suggesting improvements. When conducting idea reviews, while prioritizing the most recent results, carefully consider not only the most recent results but also all previous trials and their outcomes to provide comprehensive and well-informed feedback.


            # VQE Ansatz Constraints
            {ansatz_constraints}


            # Cost Function Definition
            {cost_function_definition}


            # Optimizer Settings
            {optimizer_settings}


            # Evaluation Priorities
            All ansatz designs must first satisfy the VQE Ansatz Constraints as fundamental prerequisites, then review every VQE ansatz in **strict descending order of importance** based on Tier order: Tier 1 has the highest priority, followed by Tier 2.


            ## Tier 1 - Gradient Viability
            **Purpose**: Ensure the optimization algorithm can compute meaningful gradients and make progress toward the energy minimum.

            - **Optimization progress verification**
                When "Number of Iterations" in user-provided feedback is 1, it indicates that the optimization process is not progressing. Review whether the circuit structure enables proper gradient computation and parameter optimization.
            - **Parameter Coverage**
                Verify that every trainable parameter appears in at least one generator whose action does **not** commute with the target Hamiltonian. If any parameter lacks such coverage, gradients will be exactly zero.
            - **Initial-State Eigencheck**
                Confirm the reference state is not an eigenstate of all generators; otherwise the initial gradient is identically zero.


            ## Tier 2 - Fundamental correctness and computational efficiency
            **Purpose**: Ensure the ansatz respects the underlying physics and chemistry of the target system while maintaining computational efficiency.

            - **Symmetry preservation**
                The circuit must conserve particle number N, total spin S^{{2}}, and relevant spatial (point-group) symmetries of the target Hamiltonian.
            - **Scaling compliance**
                Re confirm resource budgets: gate count should be O(N^2) or less and parameter count should be O(N^2) or less.
            - **Size consistency and size extensivity**
                Energy errors must not grow super-linearly with system size; the design should remain valid when the molecule is enlarged.
            - **Ease of initialization**
                Prefer designs that accept chemically meaningful starting points (e.g., Hartree-Fock angles) without ad-hoc tuning.
            - **Smoothness across geometries**
                Under a bond length scan, the optimized energy should vary smoothly without spurious discontinuities caused by ansatz induced symmetry breaking or parameter non identifiability.


            # Output Format
            1. Keep Points:
                - Identify factors that contributed to minimizing the cost function.
                - Analyze the most accurate idea in detail.
                - Review multiple ideas to identify common elements that contributed to minimizing the cost function.
            2. Suggestions:
                - Limit the number of suggestions to {max_suggestion_num} or fewer.
                - Ensure each suggestion includes only a single proposal.
                - Each suggestion should be independent and not overlap with other suggestions.
                - Prioritize the most impactful suggestions.
                - If no suggestions for improvement are identified, return suggestions: ["COMPLETED"].
            3. Output Schema: {{
                "keep_points": ["point 1", "point 2", ..., "point n"],
                "suggestions": ["suggestion 1", "suggestion 2", ..., "suggestion n"]
                }}.
            """
        )

    def _build_user_pormpt(
        self,
        last_trial_num: int,
        last_trial_results: str,
        performance_review: str | None,
    ) -> dict[str, str]:
        template = dedent(
            """
            The previous idea and experimental results are provided below in the "# Previous Trial Idea and Results" section. These reflect iterative adjustments based on past trial review comments.

            Review the trial results to design a VQE ansatz for a more accurate ground-state energy estimation. Identify the factors that contributed to minimizing the cost function and areas for further enhancement.

            # Previous Trial Idea and Results (Trial Number: {last_trial_num})
            {last_trial_results}
            """
        )

        if performance_review is not None:
            template = performance_review + template

        return {
            "role": "user",
            "content": template.format(
                last_trial_num=last_trial_num,
                last_trial_results=last_trial_results.strip(),
            ),
        }

    def build(
        self,
        last_trial_num: int,
        last_trial_results: str,
        performance_review: str | None = None,
    ) -> tuple[dict[str, str], dict[str, str]]:
        if self.target_molecule_names:
            target_molecule_statement = f"\nThe cost function is calculated based on the {self.target_molecule_names} molecules."
        else:
            target_molecule_statement = ""

        developer_content = self.developer_template.format(
            max_suggestion_num=self.max_suggestion_num,
            ansatz_constraints=self.ansatz_constraints,
            cost_function_definition=self.cost_function_definition
            + target_molecule_statement,
            optimizer_settings=self.optimizer_settings,
        )
        developer_prompt = {"role": "developer", "content": developer_content}
        user_prompt = self._build_user_pormpt(
            last_trial_num, last_trial_results, performance_review
        )

        return developer_prompt, user_prompt
