# flake8: noqa

from textwrap import dedent

SCORING_FEW_SHOTS = dedent(
    """
    - Unitary Coupled-Cluster Singles and Doubles (UCCSD):
    - explanation:
        - The **Unitary Coupled-Cluster Singles and Doubles (UCCSD)** ansatz is a quantum circuit inspired by methods in quantum chemistry for approximating ground-state wavefunctions of molecular systems. It constructs a parameterized unitary operator by exponentiating a sum of single and double excitation operators, which promote electrons from occupied to virtual orbitals. Formally, the ansatz is represented as
        $e^{T - T^{\dagger}}$ where $T$ includes terms like $T_1$ (singles) and $T_2$ (doubles). These are defined as: $T_1 = \sum_{i,a} t_i^a a_a^\dagger a_i, \quad T_2 = \sum_{i<j, a<b} t_{ij}^{ab} a_a^\dagger a_b^\dagger a_j a_i$ where $t$ coefficients are variational parameters, $a^\dagger$ and $a$ denote fermionic creation and annihilation operators, and indices $i, j$ (occupied) and $a, b$ (virtual) correspond to orbital labels. The unitary form ensures better preservation of the norm and can be naturally implemented on a quantum computer via Trotterized sequences of quantum gates. UCCSD is particularly valuable for its chemically inspired structure, allowing for physically meaningful approximations, but it tends to become resource-intensive for large molecules due to circuit depth and the number of parameters.
    - scores:
        - Originality: 7.5
        - Feasibility: 6.0
        - Versatility: 7.0

    - Hardware-Efficient Ansatz (HEA):
    - explanation:
        - The Hardware-Efficient Ansatz is designed with a focus on minimizing circuit depth and gate complexity on noisy intermediate-scale quantum (NISQ) hardware. It typically consists of alternating layers of single-qubit rotation gates (e.g., RX, RY, RZ) and entangling gates (e.g., CNOTs) arranged according to the connectivity of the quantum processor. The general form is not physically motivated by any particular problem class, which makes it highly adaptable, though potentially less expressive for certain structured quantum states. HEAs are especially effective for proof-of-concept demonstrations and optimization tasks, but they often suffer from barren plateaus in large systems.
    - scores:
        - Originality: 5.0
        - Feasibility: 9.0
        - Versatility: 8.0

    - k-fold Unitary Pair Coupled Cluster Generalized Singles and Doubles (k-UpCCGSD):
    - explanation:
        - The **k-fold Unitary Pair Coupled Cluster with Generalized Singles and Doubles (k-UpCCGSD)** ansatz approximates molecular wavefunctions by applying a restricted set of excitation operators multiple times: $\ket{\psi_{\text{k-UpCCGSD}}} = \left(\prod_{l=1}^{k} e^{\hat{T}_l - \hat{T}_l^\dagger}\right) \ket{\phi_0}$ Each \( \hat{T}_l \) includes **pair doubles** and **generalized singles**: $\hat{T}_l = \sum_{i,a} t_i^a a_a^\dagger a_i + \sum_{p<q} t_{pq} a_p^\dagger a_q^\dagger$ The parameter \( k \) controls expressibility, enabling high chemical accuracy while keeping circuit depth manageable.
    - scores:
        - Originality: 8.0
        - Feasibility: 7.0
        - Versatility: 6.0

    - Qubit Coupled Cluster (QCC):
    - explanation:
        - The **Qubit Coupled Cluster (QCC)** ansatz constructs the wavefunction in qubit space directly, avoiding second-quantized fermionic operators. It expresses the ansatz as a product of exponentials of Pauli strings: $\ket{\psi_{\text{QCC}}} = \prod_{k=1}^{N} e^{-i \theta_k P_k} \ket{\phi_0}$ where \( P_k \) is a Pauli string (e.g., \( X_1 Y_2 Z_3 \)) and \( \theta_k \) is a variational parameter. This removes the need for fermion-to-qubit mapping and can simplify circuit design, at the expense of chemical interpretability.
    - scores:
        - Originality: 8.5
        - Feasibility: 6.5
        - Versatility: 5.5

    - Factorized Unitary Coupled Cluster (fUCC):
    - explanation:
        - The **Factorized Unitary Coupled Cluster (fUCC)** method expresses the UCC ansatz as a product of unitary gates for each excitation: $\ket{\psi_{\text{fUCC}}} = \prod_{\mu} e^{\theta_\mu (\hat{\tau}_\mu - \hat{\tau}_\mu^\dagger)} \ket{\phi_0}$ Here, \( \hat{\tau}_\mu \) is an excitation operator (e.g., single or double). This factorized form avoids Trotterization and improves gate scheduling and parallelism, making it suitable for near-term hardware.
    - scores:
        - Originality: 7.0
        - Feasibility: 8.0
        - Versatility: 6.5

    - k-order Unitary Coupled Cluster (k-UCC):
    - explanation:
        - The **k-order Unitary Coupled Cluster (k-UCC)** method generalizes UCCSD by including higher-order excitations in the cluster operator:$\hat{T} = \hat{T}_1 + \hat{T}_2 + \hat{T}_3 + \cdots + \hat{T}_k$, $\ket{\psi_{\text{k-UCC}}} = e^{\hat{T} - \hat{T}^\dagger} \ket{\phi_0}$ where each \( \hat{T}_n \) represents n-body excitations (e.g., triples, quadruples). This provides better accuracy for strongly correlated systems, but becomes infeasible for current hardware due to exponential scaling.
    - scores:
        - Originality: 6.5
        - Feasibility: 3.0
        - Versatility: 4.5
    """
)
