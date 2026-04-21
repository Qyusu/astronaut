from textwrap import dedent

RESEARCHER_USER_PROMPT = dedent(
    """
    # Objective
    Conduct a comprehensive survey of existing research and knowledge to inform ansatz design for molecular ground state estimation using Variational Quantum Eigensolver (VQE). This report will be utilized as reference material for AI-driven automated ansatz design.

    # Research Scope
    **General Background**:
        - Investigate the background, current state, challenges, and general insights or recommendations for ansatz design.
    **Ansatz Comparison**:
        - Survey and compare the different ansatz architectures reported in existing research. Please compare a wide range of ansatz types, not limited to hardware-efficient ansatz.
    **Exploratory Investigation**:
        - Since VQE is still a relatively new technology, conduct a comprehensive survey focusing on quantum chemistry and extending as far back in history as possible.
        - While our ultimate goal is to design Hardware-Efficient ansatz, we seek comprehensive investigation tracing back through quantum chemistry history and related fields to explore new directions and ideas.
        - This section is most important. Please spend most of your time on this section.

    Once your review is complete, please generate a report that follows the report format below:

    # Report Format
    ## General Background
        - Background
        - Current State
        - Challenges
        - Insights & Recommendations

    ## Ansatz Comparison
        - Ansatz Name
        - Core structure or design principle
        - Advantages (Pros)
        - Limitations (Cons)
        - Suitability for iterative optimization

    ## Exploratory Investigation
        - Please provide comprehensive and easy-to-understand documentation of your research findings. There are no character limits, so please include exhaustive details in your investigation results.
    """
)
