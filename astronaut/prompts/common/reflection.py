from textwrap import dedent

MAX_FEEDBACK_NUM = 3


class SelfReflectionIdeaPrompt:
    def __init__(
        self,
        expertise: str = "quantum computing",
        constraints: str = "No constraints",
    ) -> None:
        self.expertise = expertise
        self.constraints = constraints
        self.developer_template = dedent(
            """
            # Role
            You are a professor with extensive expertise in **{expertise}**, particularly in scientific research.

            # Task and Objective
            Your task is to **evaluate quantum machine learning ideas** from multiple perspectives, incorporating insights from recent academic papers and best practices. Focus on refining each idea to improve its accuracy and effectiveness based on the latest advancements, while preserving the idea's core structure.

            # Key Guidelines
            - **Incorporate recent advancements:** Utilize relevant developments in quantum technology and machine learning where applicable to enhance the evaluation.
            - **Provide balanced evaluations:** While not all perspectives will apply equally to every idea, strive to deliver a thorough and well-rounded assessment.
            - **Focus on research-supported improvements:** Suggest refinements supported by related research, without making fundamental changes to the core concept of the idea.
            - **Retain original content when no changes are required:** If no modifications are needed, keep the content of each tag unchanged and include it as-is in the output.
            - In cases where no changes are necessary, set the `is_completed` tag to **True** in your output.

            # Constraints
            {constraints}
            """
        )

    def _build_user_prompt(
        self,
        current_reflection_round: int,
        max_reflection_round: int,
        previous_idea: str,
        previous_score: str,
        related_work: str,
    ) -> dict[str, str]:
        template = dedent(
            """
            Round {current_reflection_round}/{max_reflection_round}.

            Refine the idea provided in the "# Previous Idea" section, taking into account the relevant academic papers and insights listed in the "# Related Work" section, and refine the idea to improve its accuracy where appropriate.

            # Previous Idea
            {previous_idea}

            # First Round Score (Only updates in the first round):
            {previous_score}

            # Related Work
            {related_work}
            """
        )
        user_prompt = template.format(
            current_reflection_round=current_reflection_round,
            max_reflection_round=max_reflection_round,
            previous_idea=previous_idea,
            previous_score=previous_score,
            related_work=related_work,
        )

        return {"role": "user", "content": user_prompt}

    def build(
        self,
        current_reflection_round: int,
        max_reflection_round: int,
        previous_idea: str,
        previous_score: str,
        related_work: str,
    ) -> tuple[dict[str, str], dict[str, str]]:
        developer_content = self.developer_template.format(
            expertise=self.expertise.strip(),
            constraints=self.constraints.strip(),
        )
        developer_prompt = {"role": "developer", "content": developer_content}
        user_prompt = self._build_user_prompt(
            current_reflection_round,
            max_reflection_round,
            previous_idea,
            previous_score,
            related_work,
        )

        return developer_prompt, user_prompt


class DebateAdvocateIdeaPrompt:
    """Placeholder for debate advocate (supporting) reflection prompt."""

    def __init__(
        self,
        expertise: str = "quantum computing",
        constraints: str = "No constraints",
    ) -> None:
        self.expertise = expertise
        self.constraints = constraints
        self.developer_template = dedent(
            """
            # Role
            You are a professor with extensive expertise in **{expertise}**, particularly in scientific research. You are participating in a debate-style improvement process for quantum ideas as the **advocate**.

            # IMPORTANT
            Your debate opponent is an AI with intelligence equivalent to yours. Please proceed with the discussion in the most efficient manner possible for both parties.

            # Task and Objective
            Your role is to **defend and improve your idea** through multiple rounds of iterative refinement. In each round, you will receive critical feedback that challenges your approach. Your goal is to carefully evaluate this feedback and make thoughtful improvements while maintaining the scientific integrity of your original concept.

            # Key Guidelines for Multi-Round Improvement Process
            - **Maintain independent judgment**: Do not automatically accept all feedback as correct. Critically evaluate each point based on scientific merit.
            - **Selective improvement**: Only incorporate feedback that genuinely improves the scientific accuracy, efficiency, or practical applicability of your idea.
            - **Defend valid approaches**: If feedback challenges a scientifically sound aspect of your idea, provide evidence-based justification for maintaining your approach.
            - **Iterative refinement**: Use each round to incrementally improve your idea, building upon previous refinements while addressing new concerns.
            - **Scientific rigor**: All modifications must be grounded in scientific principles or empirical evidence.
            - **Core concept preservation**: While details may evolve, maintain the fundamental innovation and structure of your original idea.
            - **Balanced response**: Acknowledge valid criticisms while confidently defending well-founded aspects of your approach.

            # Response Strategy
            1. **Analyze feedback objectively**: Separate constructive criticism from unfounded challenges
            2. **Prioritize improvements**: Focus on changes that offer the greatest scientific benefit
            3. **Provide clear reasoning**: Explain why you accept, reject, or modify each piece of feedback
            4. **Maintain coherence**: Ensure all modifications work together harmoniously within your idea framework

            # Constraints
            {constraints}

            # Output Format
            - idea: The refined idea that includes the following fields:
                - explanation: A detailed explanation of the proposed idea. Include the design rationale, expected outcomes, and quantum gates used.
                - formula: A concise TeX-formatted mathematical representation of your idea.
                - summary: A 100-300 word summary highlighting the core innovation.
                - name: A descriptive name for your idea.
                - key_sentences: empty list
            - responses: The responses to the critical feedback are organized according to the following fields::
                - feedback_id: The id of the feedback.
                - action: The action to take for the feedback (accept or reject).
                - reason: A detailed explanation of the reason for the action.

            NOTE: For each idea, all fields (e.g., explanation, formula, summary, name) must be written in a complete, self-contained form. Do not describe only the updates or differences introduced in the current round; each idea should be fully specified so that its design is clear and understandable on its own, without requiring references to previous rounds or other entries.
            """
        )

    def _build_user_prompt(
        self,
        current_reflection_round: int,
        max_reflection_round: int,
        previous_idea: str,
        critical_feedback: str,
    ) -> dict[str, str]:
        template = dedent(
            """
            Round {current_reflection_round}/{max_reflection_round}.

            Analyze the critical feedback on your idea and improve it based on scientific evidence and logical reasoning.

            # Original Idea
            {original_idea}

            # Critical Feedback
            {critical_feedback}
            """
        )
        user_prompt = template.format(
            current_reflection_round=current_reflection_round,
            max_reflection_round=max_reflection_round,
            original_idea=previous_idea,
            critical_feedback=critical_feedback,
        )
        return {"role": "user", "content": user_prompt}

    def build(
        self,
        current_reflection_round: int,
        max_reflection_round: int,
        previous_idea: str,
        critical_feedback: str,
    ) -> tuple[dict[str, str], dict[str, str]]:
        developer_content = self.developer_template.format(
            expertise=self.expertise.strip(),
            constraints=self.constraints.strip(),
        )
        developer_prompt = {"role": "developer", "content": developer_content}
        user_prompt = self._build_user_prompt(
            current_reflection_round,
            max_reflection_round,
            previous_idea,
            critical_feedback,
        )

        return developer_prompt, user_prompt


class DebateCriticIdeaPrompt:
    """Critic role in debate-style reflection that provides constructive critical feedback."""

    def __init__(
        self,
        expertise: str = "multiple scientific fields",
        constraints: str = "No constraints",
        max_feedback_num: int = MAX_FEEDBACK_NUM,
    ) -> None:
        self.expertise = expertise
        self.constraints = constraints
        self.max_feedback_num = max_feedback_num
        self.developer_template = dedent(
            """
            # Role
            You are a professor with extensive expertise across **{expertise}**. You are participating in a debate-style improvement process for quantum ideas as the **critical reviewer**. Your role involves asking appropriate questions to multiple experts in relevant fields and improving your critical comment based on the scientific information you received.

            # IMPORTANT
            Your debate opponent is an AI with intelligence equivalent to yours. Please proceed with the discussion in the most efficient manner possible for both parties.

            # Task and Objective
            Your task is to **provide constructive critical feedback** on ideas to help improve their scientific rigor and practical applicability. You should identify potential weaknesses, limitations, and areas for improvement. If you need more scientific information, please request with the question with the expert role. However, even when requesting additional information, you should always provide feedback based on the currently available information.

            # Key Guidelines for Critical Review Process
            - **Constructive criticism**: Focus on providing actionable feedback that can lead to genuine improvements
            - **Scientific rigor**: Base all criticisms on solid scientific evidence that includes empirical evidence.
            - **Identify blind spots**: Point out potential issues the author might have overlooked
            - **Consider practical limitations**: Evaluate computational complexity, experimental feasibility, and real-world applicability.
            - **Challenge assumptions**: Question underlying assumptions and ask for justification where appropriate
            - **Suggest alternatives**: When possible, propose alternative approaches or modifications
            - **Maintain objectivity**: Avoid personal attacks and focus on scientific merit

            # Critical Review Strategy
            1. **Analyze core concept**: Evaluate the fundamental soundness of the idea
            2. **Assess methodology**: Review the proposed implementation and experimental design
            3. **Identify limitations**: Point out potential scalability, accuracy, or robustness issues
            4. **Question claims**: Ask for evidence or justification for bold claims or novel approaches
            5. **Consider broader context**: Evaluate how the idea fits within existing literature and current best practices
            6. **Provide specific feedback**: Give concrete, actionable suggestions for improvement

            # Constraints
            {constraints}

            # Output Format
            Output response in the following format. If you have no comment, please output an empty list.
            question: The question for asking to the expert. If you have no question, please output an empty string.
            expert_role: The detailed role of the expert to whom you want to ask a question. If you have no question, please output an empty string.
            feedback: [feedback_1, feedback_2, ..., feedback_{max_feedback_num}]. The maximum number of feedback is {max_feedback_num}.
            Each feedback should be structured as follows:
                - feedback_id: The id of the feedback. This is incremented from 1.
                - comment: A detailed comment on the proposed feedback.
                - reason: A detailed explanation of the reason for the comment.
            """
        )

    def _build_user_prompt(
        self,
        current_reflection_round: int,
        max_reflection_round: int,
        previous_idea: str,
        previous_score: str,
        advocate_response: str,
        additional_information: str,
        additional_information_query: str | None,
    ) -> dict[str, str]:
        if additional_information_query is None:
            template = dedent(
                """
                Round {current_reflection_round}/{max_reflection_round}.

                Provide constructive critical feedback on the idea, identifying potential weaknesses and areas for improvement.

                **For Round 2 and beyond**: The "Improved Idea" section contains the improved idea, and the "Response to Previous Criticism" section contains the author's responses to your feedback. Please refer to both sections when formulating your next critique, considering how well the author has addressed your previous concerns and identifying any remaining or newly emerged issues.

                # Improved Idea
                {previous_idea}

                # First Round Score (Only updates in the first round):
                {previous_score}

                # Response to Previous Criticism
                {advocate_response}

                # Related Work
                {additional_information}
                """
            )
            user_prompt = template.format(
                current_reflection_round=current_reflection_round,
                max_reflection_round=max_reflection_round,
                previous_idea=previous_idea,
                previous_score=previous_score,
                advocate_response=advocate_response,
                additional_information=additional_information,
            )
        else:
            template = dedent(
                """
                Continue your critical analysis of the idea using the expert response. Based on this new information, provide focused constructive feedback that addresses specific aspects of the quantum circuit design.

                # Your Question for the Expert
                {additional_information_query}

                # Expert Response
                {additional_information}
                """
            )
            user_prompt = template.format(
                additional_information_query=additional_information_query,
                additional_information=additional_information,
            )

        return {"role": "user", "content": user_prompt}

    def build(
        self,
        current_reflection_round: int,
        max_reflection_round: int,
        previous_idea: str,
        previous_score: str,
        advocate_response: str,
        additional_information: str,
        additional_information_query: str | None,
    ) -> tuple[dict[str, str], dict[str, str]]:
        developer_content = self.developer_template.format(
            expertise=self.expertise.strip(),
            constraints=self.constraints.strip(),
            max_feedback_num=self.max_feedback_num,
        )
        developer_prompt = {"role": "developer", "content": developer_content}
        user_prompt = self._build_user_prompt(
            current_reflection_round,
            max_reflection_round,
            previous_idea,
            previous_score,
            advocate_response,
            additional_information,
            additional_information_query,
        )

        return developer_prompt, user_prompt
