# flake8: noqa

from textwrap import dedent

from astronaut.constants import PENNYLANE_VERSION, REASONING_SERIES
from astronaut.prompts.pennylane_operations import get_pennylane_operations


class GenerateFeatureMapIdeaPrompt:
    def __init__(
        self,
        llm_model_version: str,
        max_trial_num: int,
        trial_num: int,
        idea_num: int,
        best_idea_abstract: str,
        device_n_qubit: int,
    ) -> None:
        self.llm_model_version = llm_model_version
        self.max_trial_num = max_trial_num
        self.trial_num = trial_num
        self.idea_num = idea_num
        self.best_idea_abstract = best_idea_abstract
        self.device_n_qubit = device_n_qubit

        self.developer_template = dedent(
            """
            You are a quantum computing expert specializing in designing Quantum Feature Maps for classification tasks using a Quantum Support Vector Machine (QSVM).

            Your task is to create a quantum feature map that will serve as the kernel function in a QSVM classifier applied to MNIST data. The ultimate goal is to design a feature map that enables the classifier to achieve high accuracy in classification.

            This task will follow an iterative improvement process, where the feature map design is refined based on review comments provided after each iteration. Use the feedback to enhance the design while maintaining alignment with the defined objectives and constraints.

            # Task Definition
            Develop multiple ideas for quantum feature maps that satisfy the following criteria. The feature maps will be used to compute the quantum kernel K(x, x') = |⟨Φ(x)|Φ(x')⟩|^2 for a QSVM. Ensure the designs are tailored to this kernel computation.
            1. **Design Considerations**:
                - Define combinations of quantum gates.
                - Define an entanglement pattern for the features.
                - Specify the method for embedding input data and quantum states as rotation angles of quantum gates.
                - Ensure the 80-dimensional input data is utilized effectively, minimizing any loss of information.
                    - Avoid excessive feature compression that may lead to information loss (e.g., simple feature averaging, summing, etc.).
            2. **Restrictions on Encoding and Embedding**:
                - Only **linear functions** are allowed for encoding and embedding.
                - All parameters in the encoding and embedding must be **non-trainable**.

            ## Key Context
            - The input data, originally represented as 784-dimensional image data, has been compressed to 80 dimensions using PCA and each value normalized to the range [0.0, 1.0].
            - Propose a quantum feature map that is independent of the number of qubits in the quantum device.

            ## Iterative Design
            - You will refine this feature map over {max_trial_num} total trials.
            - Each trial, you'll receive evaluation feedback to help evolve the design.
            - In subsequent trials, the primary goal is to improve classification accuracy based on the feedback provided, while maintaining or improving the feature map's fidelity and computational feasibility.
            
            ## Output Format
            results: [idea_1, idea_2, ..., idea_n]
            Each idea should be structured as follows:
                - explanation: A detailed explanation of the proposed feature map. Include design rationale, expected outcomes, and quantum gates used.
                - formula: A concise TeX-formatted mathematical representation of your idea.
                - summary: A 100-300 word summary highlighting the core innovation.
                - feature_map_name: A descriptive name for your feature map.
                - key_sentences: Up to 5 key sentences, each 50-100 words, that describe the essential aspects of your design for subsequent vector-based searching.

            ## Important Notes
            - Clarity is paramount. Reiterate points if necessary to ensure understanding. There is no length restriction on the explanation—ensure all relevant details are provided.
            - Evaluation is performed using an ideal quantum simulator without noise, so hardware noise does not need to be considered.
            """
        )

    def _build_user_prompt(self, review_comment: str) -> dict[str, str]:
        if self.trial_num == 1:
            template = dedent(
                """
                Trial {current_trial}/{max_trial_num}.

                This is the **first trial** of the quantum feature map design task.  
                For this initial trial:  
                - Focus on designing high-accuracy quantum feature maps while exploring diverse approaches to feature map design.
                - Ensure the designs align with the following constraints:
                    1. The idea itself must be a feature map that is independent of the number of qubits in the quantum device, but evaluation will be conducted using an {device_n_qubit}-qubit simulator.
                    2. The input data consists of **80-dimensional PCA-reduced MNIST data**, with each value normalized to the range [0.0, 1.0].
                    3. The encoding method is restricted to **non-trainable parameters** and **linear functions**.
                    4. Ensure effective utilization of the 80-dimensional input data while minimizing information loss.
                        - Avoid excessive feature compression that may lead to information loss (e.g., simple feature averaging, summing, etc.).

                ### Key Objective
                Create **{idea_num} high-accuracy quantum feature map ideas** that explore diverse directions and can serve as strong foundations for refinement in future trials.
                """
            )
            user_prompt = template.format(
                current_trial=self.trial_num,
                max_trial_num=self.max_trial_num,
                device_n_qubit=self.device_n_qubit,
                idea_num=self.idea_num,
            )

        else:
            template = dedent(
                """
                Trial {current_trial}/{max_trial_num}.

                ### Feedback from the Previous Trial (Trial {previous_trial})
                {review_comment}

                ### Task for This Trial
                Based on the feedback provided above, refine your quantum feature map ideas and generate a total of **{idea_num} improved ideas**. These ideas should aim to enhance the classification accuracy.

                ### Key Guidelines
                - The feedback includes the following:  
                    - **Keep Points:** Aspects of the previous design that should be retained.  
                    - **Suggestions:** Areas for improvement or new directions to explore.  

                - Variety in Approaches:  
                    - You are not required to address all feedback points in a single idea.
                    - Ensure that the multiple ideas generated based on the review incorporate different directions of improvement, maintaining diversity.

                ### Primary Objective
                The primary goal in this trial is to **boost classification accuracy** through iterative refinements while ensuring the designs align with the constraints.
                """
            )
            user_prompt = template.format(
                current_trial=self.trial_num,
                max_trial_num=self.max_trial_num,
                previous_trial=self.trial_num - 1,
                review_comment=review_comment.strip(),
                idea_num=self.idea_num,
            )

        return {"role": "user", "content": user_prompt}

    def build(self, review_comment: str) -> tuple[dict[str, str], dict[str, str]]:
        developer_content = self.developer_template.format(max_trial_num=self.max_trial_num)
        if self.llm_model_version in REASONING_SERIES:
            system_prompt = {"role": "developer", "content": developer_content}
        else:
            system_prompt = {"role": "system", "content": developer_content}

        user_prompt = self._build_user_prompt(review_comment)

        return system_prompt, user_prompt


class ReflectionFeatureMapIdeaPrompt:
    def __init__(self, llm_model_version: str) -> None:
        self.llm_model_version = llm_model_version
        self.developer_template = dedent(
            """
            You are a professor with extensive expertise in **quantum computing** and **machine learning**, particularly in scientific research.

            Your task is to **evaluate quantum machine learning ideas** from multiple perspectives, incorporating insights from recent academic papers and best practices. Focus on refining each idea to improve its accuracy and effectiveness based on the latest advancements, while preserving the idea's core structure.

            ### Notes
            - **Incorporate recent advancements:** Utilize relevant developments in quantum technology and machine learning where applicable to enhance the evaluation.
            - **Provide balanced evaluations:** While not all perspectives will apply equally to every idea, strive to deliver a thorough and well-rounded assessment.
            - **Design considerations:** Restrict encoding to **non-trainable parameters** and **linear functions**.
            - **Focus on research-supported improvements:** Suggest refinements supported by related research, without making fundamental changes to the core concept of the idea.
            - **Retain original content when no changes are required:** If no modifications are needed, keep the content of each tag unchanged and include it as-is in the output.
            """
        )

    def _build_user_pormpt(
        self, current_round: int, max_reflection_round: int, previous_idea: str, previous_score: str, related_work: str
    ) -> dict[str, str]:
        template = dedent(
            """
            Round {current_round}/{max_reflection_round}.

            Carefully review the idea provided in the "# Previous Idea" section, along with its score from the "# First Round Score" section. When conducting your evaluation, take into account the relevant academic papers and insights listed in the "# Related Work" section. 
            
            After your analysis and evaluation:
            - Refine and improve the idea for high-accuracy where appropriate, ensuring that the **core concept remains unchanged**.
            - If no modifications are required, retain the following tags as-is: `feature_map_name`, `summary`, `explanation`, `formula`, and `key_sentences`.
            - In cases where no changes are necessary, set the `is_completed` tag to **True** in your output.

            # Previous Idea
            {previous_idea}

            # First Round Score:
            {previous_score}

            # Related Work
            {related_work}
            """
        )
        user_prompt = template.format(
            current_round=current_round,
            max_reflection_round=max_reflection_round,
            previous_idea=previous_idea,
            previous_score=previous_score,
            related_work=related_work,
        )

        return {"role": "user", "content": user_prompt}

    def build(
        self, current_round: int, max_reflection_round: int, previous_idea: str, previous_score: str, related_work: str
    ) -> tuple[dict[str, str], dict[str, str]]:
        if self.llm_model_version in REASONING_SERIES:
            system_prompt = {"role": "developer", "content": self.developer_template}
        else:
            system_prompt = {"role": "system", "content": self.developer_template}

        user_prompt = self._build_user_pormpt(
            current_round,
            max_reflection_round,
            previous_idea,
            previous_score,
            related_work,
        )

        return system_prompt, user_prompt


class GenerateFeatureMapCodePrompt:
    def __init__(
        self, code: str, idea: str, llm_model_version: str, pennylane_version: str = PENNYLANE_VERSION
    ) -> None:
        self.code = code
        self.idea = idea
        self.llm_model_version = llm_model_version
        self.pennylane_version = pennylane_version

        self.developer_tempalte = dedent(
            """
            You are an expert Python programmer specializing in quantum computing with extensive knowledge of the PennyLane library.

            # Task Definition
            Your task is to implement ideas for quantum feature maps in Python code using the PennyLane library. The available operations in PennyLane are listed under the "# Available PennyLane Operations" section.

            # Input Data
            The `feature_map` method receives the input data `x`, which represents one sample of the dataset. This input is an 80-dimensional NumPy array with the shape `(80,)`. Each value already normalized to the range [0.0, 1.0].

            # Key Guidelines:
            1. **Base Code Format**:
                - Use the provided base code format as the template for all implementations.
                - Do not change function names or the overall structure of the base code.
                - Do not include any operations for measuring quantum states, such as qml.exp or qml.measure, etc
                - Ensure the code adheres to the format specified below:
                    ```python
                    {code}
                    ```
                - All hyperparameters (e.g., reps, c, etc.) must be defined as arguments in the __init__ method of the FeatureMap class.

            2. **Initialization Parameters**: 
                - When defining hyperparameters other than "self" and "n_qubits", always specify default values.
                - Default values should be derived from the user-provided idea.  

            3. **Imports and Libraries**:
                - Include all necessary import libraries.
                - If external libraries are required, ensure they are properly included in the code.

            4. **Code Quality**:
                - Use clear and consistent **argument names** and include **type hints** for all methods.
                - Add relevant **comments** to explain the purpose and functionality of the code.
                - Ensure that the generated code is **executable with PennyLane**.

            5. **PennyLane Operations**:
                - Limit your implementation to the operations listed under the "# Available PennyLane Operations" section.
                - Each PennyLane operation should be explicitly assigned argument names.

            # Available PennyLane Operations
            {pennylane_operations}

            ---

            # Output Format
            Implement result should be structured as follows:
                - class_name: Name of the generated feature map class
                - code: generated feature map code

            # Notes:
            - The quantum feature maps you design must strictly follow the base code schema provided.
            - Focus on creating efficient and clear implementations that align with best practices in quantum computing and software development.
            """
        )

    def _build_user_pormpt(self) -> dict[str, str]:
        template = dedent(
            """
            Please implement a quantum feature map based on the designs described in the "# Idea" section by extending the `BaseFeatureMap` class using PennyLane code. The idea is based on the assumption that it does not depend on the number of qubits. However, the generated code will be executed on a 10-qubit simulator.

            # Idea
            {idea}
            """
        )

        return {
            "role": "user",
            "content": template.format(idea=self.idea.strip()),
        }

    def build(self) -> tuple[dict[str, str], dict[str, str]]:
        developer_content = self.developer_tempalte.format(
            code=self.code, pennylane_operations=get_pennylane_operations(self.pennylane_version)
        )
        if self.llm_model_version in REASONING_SERIES:
            system_prompt = {"role": "developer", "content": developer_content}
        else:
            system_prompt = {"role": "system", "content": developer_content}

        user_prompt = self._build_user_pormpt()

        return system_prompt, user_prompt


class RetryGenerateFeatureMapCodePrompt:
    def __init__(self, code: str, error_messages: list[str], warning_messages: list[str]) -> None:
        self.code = code
        self.error_messages_string = (
            "- " + "\n- ".join(error_messages) if len(error_messages) > 0 else "Errors not found."
        )
        self.warning_messages_string = (
            "- " + "\n- ".join(warning_messages) if len(warning_messages) > 0 else "Warnings not found."
        )

        self.system_prompt = {
            "role": "system",
            "content": dedent(
                """
                You are an expert qunatum computing software engineer skilled in designing quantum feature maps using the PennyLane library in Python.

                ## Task Definition
                You are tasked with fixing the errors and warnings in the quantum feature map code provided by the user.
                Your task is only fix errors abd warnings. Do not change the logic or structure of the code.
                """
            ),
        }

        tempalte = dedent(
            """
            Please correct the following errors in the quantum feature map code.

            ## Code:
            {code}

            ## Errors
            {error_messages_string}

            ## Warnings
            {warning_messages_string}

            # Output Format
            Make sure the code follows the same structure as the base code provided and is formatted in the following JSON format:
            - class_name: Name of the generated feature map class
            - params: dictionary of parameters for the feature map class
            - code: generated feature map code
            """
        )
        self.user_prompt = {
            "role": "user",
            "content": tempalte.format(
                code=self.code,
                error_messages_string=self.error_messages_string,
                warning_messages_string=self.warning_messages_string,
            ),
        }

    def build(self) -> tuple[dict[str, str], dict[str, str]]:
        return self.system_prompt, self.user_prompt
