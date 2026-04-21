from textwrap import dedent

RESEARCHER_USER_PROMPT = dedent(
    """
    # Objective
    Conduct a comprehensive survey of existing research and knowledge to inform feature map design for image classification using Quantum Support Vector Machine (QSVM). This report will be utilized as reference material for AI-driven automated feature map design.

    # Research Scope
    **General Background**:
        - Investigate the background, current state, challenges, and general insights or recommendations for feature map design.
    **Feature Map Comparison**:
        - Survey and compare the different feature map architectures reported in existing research. Please compare a wide range of feature map types, including but not limited to image classification.
    **Exploratory Investigation**:
        - Since QSVM is still a relatively new technology, conduct a comprehensive survey focusing on classical machine learning and extending as far back in history as possible.
        - While our ultimate goal is to design high accuracy and general purpose feature map, we seek comprehensive investigation tracing back through classical and quantum machine learning history and related fields to explore new directions and ideas.
        - This section is most important. Please spend most of your time on this section.

    Once your review is complete, please generate a report that follows the report format below:

    # Report Format
    ## General Background
        - Background
        - Current State
        - Challenges
        - Insights & Recommendations

    ## Feature Map Comparison
        - Feature Map Name
        - Core structure or design principle
        - Advantages (Pros)
        - Limitations (Cons)
        - Suitability for iterative optimization

    ## Exploratory Investigation
        - Please provide comprehensive and easy-to-understand documentation of your research findings. There are no character limits, so please include exhaustive details in your investigation results.
    """
)
