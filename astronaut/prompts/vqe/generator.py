# flake8: noqa

from textwrap import dedent

from astronaut.prompts.vqe.constraints import (
    ANSATZ_CONSTRAINTS,
    COST_FUNCTION_DEFINITION,
    MOLECULE_HAMILTONIAN_DEFINITION,
    OPTIMIZER_SETTINGS,
)


class BaseGenerateAnsatzIdeaPrompt:
    def __init__(
        self,
        ansatz_constraints: str = ANSATZ_CONSTRAINTS.strip(),
        molecule_hamiltonian_definition: str = MOLECULE_HAMILTONIAN_DEFINITION.strip(),
        cost_function_definition: str = COST_FUNCTION_DEFINITION.strip(),
        optimizer_settings: str = OPTIMIZER_SETTINGS.strip(),
        target_molecule_names: str = "",
    ) -> None:
        self.ansatz_constraints = ansatz_constraints
        self.molecule_hamiltonian_definition = molecule_hamiltonian_definition
        self.cost_function_definition = cost_function_definition
        self.optimizer_settings = optimizer_settings
        self.target_molecule_names = target_molecule_names

        self.developer_template = dedent(
            """
            # Role
            You are an expert in quantum physics and quantum chemistry computation, specializing in variational quantum eigensolver (VQE) ansatz design.


            # Task and Objective
            Your task is to create a quantum ansatz for VQE calculations with the objective to **minimize the cost function value while satisfying the constraints**. Ideas are improved incrementally through multiple trials and their feedback results. Your strategy should adapt based on the user's instructions:

            - **Exploration strategy**: When instructed to explore, or during early trials, conduct broad exploration across diverse architecture-level approaches to identify promising directions.
            - **Refinement strategy**: When instructed to refine, use feedback from previous trials to gradually narrow idea diversity and focus on specific improvements to achieve high performance.

            Follow the user's explicit instructions regarding which strategy to employ for the current trial. After generating ideas, please ensure that you verify whether the constraint conditions are satisfied.

            # VQE Ansatz Constraints
            {ansatz_constraints}


            # Hamiltonian
            {molecule_hamiltonian_definition}


            # Cost Function Definition
            {cost_function_definition}


            # Optimizer Settings
            {optimizer_settings}


            # Experiment Feedback Guidelines
            - The feedback includes the following:
                - **Keep Points:** Aspects of the previous design that should be retained.
                - **Suggestions:** Areas for improvement or new directions to explore.
            - Variety in Approaches:
                - You are not required to address all feedback points in a single idea.
                - Ensure that the multiple ideas generated based on the review incorporate different directions of improvement, maintaining diversity.


            # Output Format
            results: [idea_1, idea_2, ..., idea_n]
            Each idea should be structured as follows:
                - explanation: A detailed explanation of the proposed ansatz. Include the design rationale, expected outcomes, and quantum gates used.
                - formula: A concise TeX-formatted mathematical representation of your idea.
                - summary: A 100-300 word summary highlighting the core innovation.
                - name: A descriptive name for your ansatz.
                - key_sentences: Up to 5 key sentences, each 50-100 words, that describe the essential aspects of your design for subsequent vector-based searching.


            # Important Notes
            - Clarity is paramount. Reiterate points if necessary to ensure understanding. There is no length restriction on the explanation—ensure all relevant details are provided.
            - The explanation of the trainable parameters is very important. Please provide a detailed information for implementation.
            """
        )

    def _build_developer_prompt(self) -> dict[str, str]:
        if self.target_molecule_names:
            target_molecule_statement = f"\nThe cost function is calculated based on the {self.target_molecule_names} molecules."
        else:
            target_molecule_statement = ""

        developer_content = self.developer_template.format(
            ansatz_constraints=self.ansatz_constraints,
            molecule_hamiltonian_definition=self.molecule_hamiltonian_definition,
            cost_function_definition=self.cost_function_definition
            + target_molecule_statement,
            optimizer_settings=self.optimizer_settings,
        )

        return {"role": "developer", "content": developer_content}


class ResearchBasedAnsatzIdeaPrompt(BaseGenerateAnsatzIdeaPrompt):
    def _build_user_prompt(
        self, idea_num: int, research_report: str
    ) -> dict[str, str]:
        template = dedent(
            """
            Based on the comprehensive research report provided below, generate **{idea_num} diverse and innovative quantum ansatz ideas** for VQE calculations.

            Please ensure that your ideas are innovative and not simply straightforward combinations of established methods from existing research. While it is perfectly acceptable to evolve and build upon existing techniques, aim for creative and novel approaches that push beyond conventional frameworks.

            In this stage, adopt the **Exploration strategy**: focus on exploring a wide range of ideas based on the research report to determine future improvement directions, rather than aiming for high performance.

            # Research Report
            {research_report}
            """
        )
        user_prompt = template.format(
            idea_num=idea_num,
            research_report=research_report.strip(),
        )
        return {"role": "user", "content": user_prompt}

    def build(
        self, idea_num: int, research_report: str
    ) -> tuple[dict[str, str], dict[str, str]]:
        developer_prompt = self._build_developer_prompt()
        user_prompt = self._build_user_prompt(idea_num, research_report)
        return developer_prompt, user_prompt


class FilteringAnsatzIdeaPrompt(BaseGenerateAnsatzIdeaPrompt):
    def _build_user_prompt(
        self, idea_num: int, experiment_report: str
    ) -> dict[str, str]:
        template = dedent(
            """
            Based on the experiment report provided below, generate **{idea_num} improved quantum ansatz ideas** for VQE calculations. When generating ideas, analyze the success and failure of the previous ideas that provided in the report.

            When referencing previous ideas in your explanation, please avoid using numbered references like 'Idea 1' and instead provide specific quoted content from the referenced ideas.

            In this stage, adopt the **Exploration strategy**: focus on identifying potential for future improvements and enhancing the comprehensiveness of exploration directions.

            # Experiment Report
            {experiment_report}
            """
        )
        user_prompt = template.format(
            idea_num=idea_num,
            experiment_report=experiment_report.strip(),
        )
        return {"role": "user", "content": user_prompt}

    def build(
        self, idea_num: int, experiment_report: str
    ) -> tuple[dict[str, str], dict[str, str]]:
        developer_prompt = self._build_developer_prompt()
        user_prompt = self._build_user_prompt(idea_num, experiment_report)
        return developer_prompt, user_prompt


class FeedbackBasedAnsatzIdeaPrompt(BaseGenerateAnsatzIdeaPrompt):
    def _no_feedback_user_prompt(
        self, current_trial_num: int, max_trial_num: int, idea_num: int
    ) -> str:
        template = dedent(
            """
            Trial {current_trial_num}/{max_trial_num}.

            Generate **{idea_num} initial ideas** for quantum ansatz design for VQE.
            """
        )
        user_prompt = template.format(
            current_trial_num=current_trial_num,
            max_trial_num=max_trial_num,
            idea_num=idea_num,
        )

        return user_prompt

    def _with_feedback_user_prompt(
        self,
        current_trial_num: int,
        max_trial_num: int,
        idea_num: int,
        review_comment: str,
        seed_ideas: str | None,
    ) -> str:
        template = dedent(
            """
            Trial {current_trial_num}/{max_trial_num}.

            Based on the feedback from the previous trial, refine your quantum ansatz ideas for VQE and generate a total of **{idea_num} improved ideas**.

            In this stage, adopt the **Refinement strategy**: through feedback from multiple trials, gradually narrow the exploration scope to generate high performance ansatz ideas. If you identify necessary improvements beyond the comments provided in the feedback, please incorporate them into your ideas.

            ## Feedback from the Previous Trial
            {review_comment}
            """
        )
        user_prompt = template.format(
            current_trial_num=current_trial_num,
            max_trial_num=max_trial_num,
            idea_num=idea_num,
            review_comment=review_comment.strip(),
        )

        if seed_ideas:
            user_prompt += f"\n\n# Seed Ideas\nThis is first trial, so provided feedback is below mutiple seed ideas:\n{seed_ideas}"

        return user_prompt

    def _build_user_prompt(
        self,
        current_trial_num: int,
        max_trial_num: int,
        idea_num: int,
        review_comment: str,
        seed_ideas: str | None,
    ) -> dict[str, str]:
        if review_comment == "":
            # This pattern only used in first trial when research_mode == "none"
            user_prompt = self._no_feedback_user_prompt(
                current_trial_num, max_trial_num, idea_num
            )
        else:
            user_prompt = self._with_feedback_user_prompt(
                current_trial_num,
                max_trial_num,
                idea_num,
                review_comment,
                seed_ideas,
            )

        return {"role": "user", "content": user_prompt}

    def build(
        self,
        current_trial_num: int,
        max_trial_num: int,
        idea_num: int,
        review_comment: str,
        seed_ideas: str | None = None,
    ) -> tuple[dict[str, str], dict[str, str]]:
        developer_prompt = self._build_developer_prompt()
        user_prompt = self._build_user_prompt(
            current_trial_num,
            max_trial_num,
            idea_num,
            review_comment,
            seed_ideas,
        )
        return developer_prompt, user_prompt


class GenerateAnsatzCodePrompt:
    def __init__(
        self,
        code: str,
        idea: str,
        pennylane_reference: str,
    ) -> None:
        self.code = code
        self.idea = idea
        self.pennylane_reference = pennylane_reference

        self.developer_template = dedent(
            """
            # Role
            You are an expert Python programmer specializing in quantum computing with extensive knowledge of the PennyLane library.


            # Task
            Your task is to implement quantum ansatz idea in Python using the PennyLane library.


            # Optimized Variables
            The `circuit` method receives the optimized variables `params`, which represent the trainable parameters of the ansatz. Defining the initialization of these trainable parameters is outside the scope of this task. However, initializing quantum states used in the ansatz (e.g., preparing reference states or applying initial quantum gates) is within the scope and should be implemented as needed.


            # Key Guidelines:
            1. **Base Code Format**:
                - Use the provided base code format as the template for all implementations. Ensure the code adheres to the format specified below:
                    ```python
                    {code}
                    ```
                - Class name `BaseAnsatz` is updated to the name of the proposed ansatz that is provided by the user.
                - Do NOT change method names or the overall structure of the base code.
                - Do NOT include any operations for measuring quantum states, such as `qml.exp` or `qml.measure`, etc.
                - All hyperparameters must be defined as arguments in the `__init__` method.
                - Note: The `params` argument in the `circuit` method is directly updated by the optimization library. Do NOT reassign or modify `params` itself. When using the parameters, access them by their index (e.g., `params[0]`, `params[1]`, ...).

            2. **Class Initialization Parameters**:
                - When defining hyperparameters other than "self", "device", and "hamiltonian", always set default values.
                - If the default value becomes complex, define it as None and set it in the `__init__` method. In this case, instead of raising an error, set an initial value that is appropriate for the design of the idea.
                - When using default values, ensure the code accurately reflects the idea without oversimplification.

            3. **Calculate Total Number of Parameters in the Ansatz**:
                - "n_params" variable represents the total number of trainable parameters in the ansatz. It is the sum of the number of parameters in all layers of the ansatz.
                - The "n_params" value must be defined in the `__init__` method.

            4. **Molecular Hamiltonian**:
                - **CRITICAL WARNING**: When accessing `MolecularHamiltonian` attributes, you MUST NEVER access them directly. You MUST ONLY use the methods explicitly listed below. Using any other attributes or methods will cause runtime errors and code failure.
                - **STRICT REQUIREMENT**: The `MolecularHamiltonian` object has ONLY the following methods available. Any attempt to access other methods (such as `.n_qubits`, `.electrons`, `.molecular_orbitals`, etc. as attributes) will result in AttributeError.
                - **ALLOWED METHODS ONLY** (these are the ONLY methods you can use):
                    - get_n_qubits(): The minimum number of qubits required to represent the hamiltonian. This is NOT the available number of qubits in the quantum device
                    - get_electrons(): The total number of electrons
                    - get_molecular_orbitals(): The number of molecular orbitals
                    - get_spin_orbitals(): The number of spin orbitals
                    - get_active_electrons(): The number of active electrons
                    - get_active_orbitals(): The number of active orbitals
                    - get_active_spin_orbitals(): The number of active spin orbitals
                - **ENFORCEMENT**: Your code will be tested and any violation of this rule will cause immediate failure. Always use the get_* methods as shown above.

            5. **Imports and Libraries**:
                - Include all necessary import libraries.
                - If external libraries are required, ensure they are properly included in the code.

            6. **Code Quality**:
                - Use clear and consistent argument names with type hints for all methods.
                - Add relevant comments to explain the purpose and functionality of the code.
                - Ensure the generated code is executable with PennyLane and follows best practices in quantum computing and software development.


            # Output Format
            Implement result should be structured as follows:
                - class_name: The name of the generated ansatz class
                - code: The generated ansatz code
            """
        )

    def _build_user_prompt(self) -> dict[str, str]:
        template = dedent(
            """
            Please implement a quantum ansatz based on the design described in the "# Idea" section. The ansatz should be independent of the number of qubits. Additionally, set default values for all required arguments when creating an instance and implement initialization processing as needed. Set the default parameters to values expected to yield high accuracy by leveraging the specific characteristics and design principles of your ansatz.

            # Idea
            {idea}

            # PennyLane Reference
            When implementing code, each PennyLane operation should be explicitly assigned argument names.
            {pennylane_reference}
            """
        )

        return {
            "role": "user",
            "content": template.format(
                idea=self.idea.strip(),
                pennylane_reference=self.pennylane_reference,
            ),
        }

    def build(self) -> tuple[dict[str, str], dict[str, str]]:
        developer_prompt = {
            "role": "developer",
            "content": self.developer_template.format(code=self.code),
        }
        user_prompt = self._build_user_prompt()

        return developer_prompt, user_prompt
