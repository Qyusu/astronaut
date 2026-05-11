# flake8: noqa

from textwrap import dedent

from astronaut.prompts.quantum_kernel.constraints import (
    FEATURE_MAP_CONSTRAINTS,
    INPUT_DATA_CONSTRAINTS,
)


class BaseGenerateFeatureMapIdeaPrompt:
    def __init__(
        self,
        feature_map_constraints: str = FEATURE_MAP_CONSTRAINTS.strip(),
        input_data_constraints: str = INPUT_DATA_CONSTRAINTS.strip(),
    ) -> None:
        self.feature_map_constraints = feature_map_constraints
        self.input_data_constraints = input_data_constraints

        self.developer_template = dedent(
            """
            # Role
            You are a quantum computing expert specializing in designing Quantum Feature Maps for classification tasks using a Quantum Support Vector Machine (QSVM).


            # Task and Objective
            Your task is to create a quantum feature map that will serve as the kernel function in a QSVM classifier, with the objective to **maximize the classification accuracy**. Ideas are improved incrementally through multiple trials and their feedback results. Your strategy should adapt based on the user's instructions:

            - **Exploration strategy**: When instructed to explore, or during early trials, conduct broad exploration across diverse architecture-level approaches to identify promising directions.
            - **Refinement strategy**: When instructed to refine, use feedback from previous trials to gradually narrow idea diversity and focus on specific improvements to achieve high performance.

            Follow the user's explicit instructions regarding which strategy to employ for the current trial. After generating ideas, please ensure that you verify whether the constraint conditions are satisfied.


            # Feature Map Constraints
            {feature_map_constraints}


            # Input Data Constraints
            {input_data_constraints}


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
                - explanation: A detailed explanation of the proposed feature map. Include design rationale, expected outcomes, and quantum gates used.
                - formula: A concise TeX-formatted mathematical representation of your idea.
                - summary: A 100-300 word summary highlighting the core innovation.
                - name: A descriptive name for your feature map.
                - key_sentences: Up to 5 key sentences, each 50-100 words, that describe the essential aspects of your design for subsequent vector-based searching.


            # Important Notes
            - Clarity is paramount. Reiterate points if necessary to ensure understanding. There is no length restriction on the explanation—ensure all relevant details are provided.
            """
        )

    def _build_developer_prompt(self) -> dict[str, str]:
        developer_content = self.developer_template.format(
            feature_map_constraints=self.feature_map_constraints,
            input_data_constraints=self.input_data_constraints,
        )
        return {"role": "developer", "content": developer_content}


class ResearchBasedFeatureMapIdeaPrompt(BaseGenerateFeatureMapIdeaPrompt):
    def _build_user_prompt(
        self, idea_num: int, research_report: str
    ) -> dict[str, str]:
        template = dedent(
            """
            Based on the comprehensive research report provided below, generate **{idea_num} diverse and innovative quantum feature map ideas** for QSVM.

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


class FilteringFeatureMapIdeaPrompt(BaseGenerateFeatureMapIdeaPrompt):
    def _build_user_prompt(
        self, idea_num: int, experiment_report: str
    ) -> dict[str, str]:
        template = dedent(
            """
            Based on the experiment report provided below, generate **{idea_num} improved quantum feature map ideas** for QSVM. When generating ideas, analyze the success and failure of the previous ideas that provided in the report.

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


class FeedbackBasedFeatureMapIdeaPrompt(BaseGenerateFeatureMapIdeaPrompt):
    def _no_feedback_user_prompt(
        self, current_trial_num: int, max_trial_num: int, idea_num: int
    ) -> str:
        template = dedent(
            """
            Trial {current_trial_num}/{max_trial_num}.

            Generate **{idea_num} initial ideas** for quantum feature map design for QSVM.
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

            Based on the feedback from the previous trial, refine your quantum feature map ideas for QSVM and generate a total of **{idea_num} improved ideas**.

            In this stage, adopt the **Refinement strategy**: through feedback from multiple trials, gradually narrow the exploration scope to generate high performance feature map ideas. If you identify necessary improvements beyond the comments provided in the feedback, please incorporate them into your ideas.

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
            user_prompt += f"\n\n# Seed Ideas\nThis is first trial, so provided feedback is below multiple seed ideas:\n{seed_ideas}"

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


class GenerateFeatureMapCodePrompt:
    def __init__(
        self,
        code: str,
        idea: str,
        pennylane_reference: str,
        input_data_constraints: str = INPUT_DATA_CONSTRAINTS.strip(),
    ) -> None:
        self.code = code
        self.idea = idea
        self.pennylane_reference = pennylane_reference
        self.input_data_constraints = input_data_constraints

        self.developer_template = dedent(
            f"""
            # Role
            You are an expert Python programmer specializing in quantum computing with extensive knowledge of the PennyLane library.

            # Task
            Your task is to implement ideas for quantum feature maps in Python code using the PennyLane library. The implementation should be independent of the number of qubits and work dynamically with any qubit count.

            # Input Data
            The `feature_map` method receives the input data `x`, which represents one sample as a NumPy array.
            {input_data_constraints}

            # Key Guidelines:
            1. **Base Code Format**:
                - Use the provided base code format as the template for all implementations. Ensure the code adheres to the format specified below:
                    ```python
                    {code}
                    ```
                - Do not change method names or the overall structure of the base code.
                - Do not include any operations for measuring quantum states, such as qml.exp or qml.measure, etc
                - Implement the feature map to work with any number of qubits. The implementation should be flexible and scalable to handle different qubit counts dynamically.

            2. **Initialization Parameters**:
                - **Core Parameters Only**: Only include essential parameters that significantly impact model performance and require hyperparameter tuning in the __init__ method:
                    * Parameters controlling circuit depth or repetition counts
                    * Parameters controlling encoding strength or coefficients
                    * Key parameters that determine quantum circuit structure
                    * Parameters that need to be adjusted based on dataset or problem settings
                - Non-essential numerical constants and auxiliary settings should be defined as fixed values within the class body, not as __init__ arguments.
                - When defining hyperparameters other than "self" and "n_qubits", always set default values. If the default value becomes complex, define it as None and set it in the __init__ method. In this case, instead of raising an error, set an initial value that is appropriate for the design of the idea.

            3. **Imports and Libraries**:
                - Include all necessary import libraries.
                - If external libraries are required, ensure they are properly included in the code.

            4. **Code Quality**:
                - Use clear and consistent **argument names** and include **type hints** for all methods.
                - Add relevant **comments** to explain the purpose and functionality of the code.
                - Ensure that the generated code is **executable with PennyLane**.

            # Output Format
            Implement result should be structured as follows:
                - class_name: Name of the generated feature map class
                - code: generated feature map code

            # Notes:
            - The quantum feature maps you design must strictly follow the base code format provided.
            - Focus on creating efficient and clear implementations that align with best practices in quantum computing and software development.
            """
        )

    def _build_user_prompt(self) -> dict[str, str]:
        template = dedent(
            """
            Please implement a quantum feature map based on the designs described in the "# Idea" section by extending the `BaseFeatureMap` class using PennyLane code. The idea is based on the assumption that it does not depend on the number of qubits, but performance evaluation will be conducted on a 10-qubit noiseless simulator (i.e., n_qubits = 10). In addition, please set default values for all arguments required when creating an instance, and implement initialization processing as needed. Set the default parameters to values expected to yield high accuracy by leveraging the specific characteristics and design principles of your feature map.

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
        developer_content = self.developer_template.format(
            code=self.code,
            input_data_constraints=self.input_data_constraints,
        )
        developer_prompt = {"role": "developer", "content": developer_content}
        user_prompt = self._build_user_prompt()

        return developer_prompt, user_prompt
