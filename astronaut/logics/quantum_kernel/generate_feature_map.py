import time
from pathlib import Path
from textwrap import dedent

import pdfplumber
import tiktoken
from langsmith import traceable
from loguru import logger
from pinecone import QueryResponse

from astronaut.configs import settings
from astronaut.constants import NOT_PROVIDED_INFORMATION
from astronaut.db import PineconeClient
from astronaut.llm import ChatClient
from astronaut.logics.common import (
    GenerateCode,
    GenerateIdea,
    ReflectIdea,
    ScoringIdea,
    SummaryPaper,
)
from astronaut.prompts.quantum_kernel import (
    GenerateFeatureMapCodePrompt,
    GenerateFeatureMapIdeaPrompt,
    ReflectionFeatureMapIdeaPrompt,
)
from astronaut.schema import (
    MESSAGE_HISTORY_TYPE,
    GeneratedIdea,
    GeneratedIdeaResult,
    GeneratedImpl,
    GeneratedImplResult,
    GeneratedResult,
    GeneratedResults,
    IdeaScore,
    MessageHistory,
    ModelVersions,
    RunContext,
    ScoringResult,
)

LOCAL_PAPER_DIR = settings.LOCAL_PAPER_DIR
MAX_PAPER_CONTENT_TOKENS = 100000
MAX_PAPER_SUMMARY_WORDS = 1000
IDEA_IMPROVEMENT_THRESHOLD = -5.0


def cut_string_if_over_token_limit(
    text: str,
    max_tokens: int,
    model_name: str = "gpt-4o",
) -> str:
    encoding = tiktoken.encoding_for_model(model_name)
    tokens = encoding.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]

    return encoding.decode(tokens)


def format_fetch_paper(result: QueryResponse) -> list[str]:
    papers = []
    for r in result["matches"]:
        paper = dedent(
            """
            Paper Id: {id}
            Abstract: {abstract}
            Releted Chunk: {chunk}
            """
        ).format(id=r["id"], abstract=r["metadata"]["abstract"], chunk=r["metadata"]["chunk_text"])
        papers.append(paper)
    return papers


def load_full_text_from_local(paper_id: str) -> str:
    if LOCAL_PAPER_DIR is None:
        raise ValueError("Local paper directory is not provided.")

    local_pdf_path = Path(f"{LOCAL_PAPER_DIR}/{paper_id}.pdf")
    if not local_pdf_path.exists():
        logger.error(f"Local PDF file is not found: {local_pdf_path}")
        return ""

    text_content = []
    with pdfplumber.open(local_pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                text_content.append(text)

    full_text = "\n".join(text_content)
    return full_text


def load_full_text(
    result: QueryResponse, max_paper_tokens: int | None = None, summarizer: SummaryPaper | None = None
) -> tuple[list[str], float]:
    summary_cost = 0.0
    papers = []
    for r in result["matches"]:
        paper = load_full_text_from_local(r["metadata"]["document_id"])
        paper = cut_string_if_over_token_limit(paper, max_paper_tokens) if max_paper_tokens is not None else paper
        if summarizer is not None:
            summary, cost = summarizer.summary(paper_content=paper, max_summary_words=MAX_PAPER_SUMMARY_WORDS)
            summary_cost += cost
            papers.append(str(summary))
        else:
            papers.append(paper)

    return papers, summary_cost


def scoring_idea(
    arxiv_db_client: PineconeClient | None,
    idea_scorer: ScoringIdea,
    idea: GeneratedIdea,
    max_scoring_round: int = 3,
    max_paper_per_query: int = 1,
    score_histories: str = "",
) -> tuple[ScoringResult, float]:
    total_cost = 0.0
    key_sentences = list(idea.key_sentences)
    history: MESSAGE_HISTORY_TYPE = []
    seen_papers: list[str] = []
    for i in range(1, max_scoring_round + 1):
        if arxiv_db_client is not None:
            # Get related work information
            related_works: list[str] = []
            for query in key_sentences:
                result = arxiv_db_client.query(
                    query, top_k=max_paper_per_query, metadata_filter={"id": {"$nin": seen_papers}}
                )
                papers = format_fetch_paper(result)
                related_works.extend(papers)
                seen_papers.extend([r["id"] for r in result["matches"]])
            related_works_str = "\n--------------".join(related_works)
        else:
            related_works_str = NOT_PROVIDED_INFORMATION

        # Evaluate the generated idea and assign the score
        scoring_result, history, cost = idea_scorer.score(
            idea=idea,
            related_work=related_works_str,
            message_history=history,
            n_history=None,
            round=i,
            max_round=max_scoring_round,
            score_histories=score_histories,
        )
        total_cost += cost

        if scoring_result.is_lack_information:
            logger.info("Lack of information. Retry to get related work information.")
            key_sentences = list(scoring_result.additional_key_sentences)
        else:
            break

    return scoring_result, total_cost


def reflect_idea(
    llm_client: ChatClient,
    arxiv_db_client: PineconeClient | None,
    model_versions: ModelVersions,
    n_idea_history: int | None,
    idea_scorer: ScoringIdea,
    seed_idea: GeneratedIdea,
    seed_score: IdeaScore,
    max_reflection_round: int,
    score_histories: str,
    max_paper_per_query: int = 3,
    summarize_paper: bool = True,
) -> tuple[GeneratedIdea, IdeaScore, float]:
    total_cost = 0.0
    final_idea = seed_idea
    final_score = seed_score
    summarizer = SummaryPaper(llm_client, model_versions.summary) if summarize_paper else None
    reflector = ReflectIdea(llm_client, model_versions.reflection, model_versions.parser)
    reflection_message_history: MESSAGE_HISTORY_TYPE = []
    for i in range(1, max_reflection_round + 1):
        logger.info(f"Reflect Idea ({i}/{max_reflection_round})...")

        if arxiv_db_client is not None:
            # get related paper and summarize by LLM
            result = arxiv_db_client.query(final_idea.summary, top_k=max_paper_per_query)
            papers, summary_cost = load_full_text(result, MAX_PAPER_CONTENT_TOKENS, summarizer)
            total_cost += summary_cost
            related_works_str = "\n--------------".join(papers)
        else:
            related_works_str = NOT_PROVIDED_INFORMATION

        # reflect the idea utilizing the related work
        system_prompt, user_prompt = ReflectionFeatureMapIdeaPrompt(llm_model_version=model_versions.reflection).build(
            current_round=i,
            max_reflection_round=max_reflection_round,
            previous_idea=str(final_idea),
            previous_score=str(final_score),
            related_work=related_works_str,
        )
        reflected_idea, reflection_message_history, cost = reflector.reflect(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            message_history=reflection_message_history,
            n_history=n_idea_history,
        )
        total_cost += cost

        # check improvement of the idea
        # reflected_idea_score, cost = scoring_idea(
        #     arxiv_db_client, idea_scorer, reflected_idea.result, score_histories=score_histories
        # )
        # total_cost += cost

        # update final idea status
        final_idea = reflected_idea.result
        # final_score = reflected_idea_score.score

        if reflected_idea.is_completed:
            logger.info("The idea reflection is completed.")
            break

    reflected_idea_score, cost = scoring_idea(
        arxiv_db_client, idea_scorer, reflected_idea.result, score_histories=score_histories
    )
    total_cost += cost
    final_score = reflected_idea_score.score

    return final_idea, final_score, total_cost


def generate_code(
    llm_client: ChatClient,
    model_versions: ModelVersions,
    last_code: str,
    finalized_idea: GeneratedIdeaResult,
    message_history: MESSAGE_HISTORY_TYPE,
    n_code_history: int | None = None,
) -> tuple[GeneratedImplResult, MESSAGE_HISTORY_TYPE, float]:
    code_generator = GenerateCode(llm_client, model_versions.code, model_versions.parser)

    generated_results = []
    for i, idea_result in enumerate(finalized_idea.results):
        logger.info(f"Generate Feature Map Code ({i+1}/{len(finalized_idea.results)})...")
        system_prompt, user_prompt = GenerateFeatureMapCodePrompt(
            code=last_code,
            idea=idea_result.get_string_for_code_generation(),
            llm_model_version=model_versions.code,
        ).build()

        for _ in range(3):
            try:
                generate_code_result, code_message_history, cost = code_generator.generate(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    message_history=message_history,
                    n_history=n_code_history,
                )
                break
            except Exception as e:
                logger.info(f"Raise Exception: {e}")
                logger.info("Retry to generate code...")
                time.sleep(3)

        message_history = code_message_history
        generated_results.append(generate_code_result)

    return GeneratedImplResult(results=generated_results), code_message_history, cost


def format_batch_results(
    ideas: list[GeneratedIdea], scores: list[IdeaScore], implements: list[GeneratedImpl]
) -> GeneratedResults:
    results = []
    for i in range(len(ideas)):
        result = GeneratedResult(idea=ideas[i], score=scores[i], implement=implements[i])
        results.append(result)
    return GeneratedResults(results=results)


@traceable(tags=["generation"])
def generate_feature_map(
    llm_client: ChatClient,
    arxiv_db_client: PineconeClient | None,
    trial_num: int,
    context: RunContext,
) -> tuple[GeneratedResults, MessageHistory, float]:
    generate_cost = 0.0

    # Generate feature map idea
    idea_prompt_generator = GenerateFeatureMapIdeaPrompt(
        llm_model_version=context.model_versions.idea,
        max_trial_num=context.max_trial_num,
        trial_num=trial_num,
        idea_num=context.max_idea_num,
        best_idea_abstract=context.best_idea_abstract,
        device_n_qubit=context.n_qubits,
    )
    idea_generator = GenerateIdea(
        client=llm_client, model_version=context.model_versions.idea, parser_model_version=context.model_versions.parser
    )
    system_prompt, user_prompt = idea_prompt_generator.build(str(context.review_comment))
    first_generated_idea, idea_map_message_history, cost = idea_generator.generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        message_history=context.message_history.idea,
        n_history=context.n_message_history.idea,
    )
    generate_cost += cost

    finalized_ideas = []
    finalized_scores = []
    for idea in first_generated_idea.results:
        # Evaluate the generated idea and assign the score
        idea_scorer = ScoringIdea(llm_client, context.model_versions.scoring)
        first_generated_idea_score, cost = scoring_idea(
            arxiv_db_client, idea_scorer, idea, score_histories=context.score_histories
        )

        # Reflect the generated idea to the code generation
        if context.max_reflection_round > 0:
            final_idea, final_score, cost = reflect_idea(
                llm_client=llm_client,
                arxiv_db_client=arxiv_db_client,
                model_versions=context.model_versions,
                n_idea_history=context.n_message_history.idea,
                idea_scorer=idea_scorer,
                seed_idea=idea,
                seed_score=first_generated_idea_score.score,
                max_reflection_round=context.max_reflection_round,
                score_histories=context.score_histories,
            )
            generate_cost += cost
        else:
            final_idea = idea
            final_score = first_generated_idea_score.score

        finalized_ideas.append(final_idea)
        finalized_scores.append(final_score)

    # Generate feature map code
    generate_code_result, code_message_history, cost = generate_code(
        llm_client=llm_client,
        model_versions=context.model_versions,
        last_code=context.last_code,
        finalized_idea=GeneratedIdeaResult(results=finalized_ideas),
        message_history=context.message_history.code,
        n_code_history=context.n_message_history.code,
    )
    generate_cost += cost

    # Format the result
    updated_message_history = MessageHistory(
        review=context.message_history.review, idea=idea_map_message_history, code=code_message_history
    )

    finalized_results = format_batch_results(finalized_ideas, finalized_scores, generate_code_result.results)

    return finalized_results, updated_message_history, generate_cost
