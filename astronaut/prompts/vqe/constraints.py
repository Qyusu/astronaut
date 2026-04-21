from textwrap import dedent

REFLECTION_EXPERTISE = "quantum chemistry computation and machine learning"

ANSATZ_CONSTRAINTS = dedent(
    """The VQE ansatz |\\Psi(\\theta)\\rangle will be used to approximate the ground state of a given Hamiltonian by variational minimization of \\langle\\Psi(\\theta)|H|\\Psi(\\theta)\\rangle.

    1. **Ansatz Design Requirements**:
        - Define the sequence and combination of parameterized quantum gates (e.g., single-qubit rotations, controlled-NOT layers).
        - All parameters must be trainable, real-valued rotation angles.
        - Specify the entanglement pattern (e.g., linear chain, full connectivity, pairwise entanglers) and the layering depth to balance expressibility and trainability.
        - **CRITICAL: Design a hardware-efficient ansatz. The total count of single-qubit rotation gates and CNOT gates scale as O(N^2) or lower, and the number of tunable parameters should also scale as O(N^2) or lower (N denotes the number of qubits).**

    2. **Execution Environment**:
        - Ansatz evaluations are performed on ideal quantum simulators without noise, so hardware noise and qubit placement constraints do not need to be considered.
        - Propose a quantum ansatz that is independent of the number of qubits in the quantum device.

    ## Hyperparameter Design Guidelines
    When designing ansatz architectures, follow these principles for hyperparameter management:

    Note: Hyperparameters refer to the parameters defined in the `__init__` method of the ansatz Python class, which determine the structure and behavior of the quantum circuit before training begins.

    - **Automatic calculation**: Ansatz hyperparameters should be automatically computed from the Hamiltonian and molecular properties whenever possible to minimize user configuration burden
    - **User-configurable parameters**: If user-settable hyperparameters are necessary, explicitly specify:
        - **Purpose**: Clear explanation of what the parameter controls and its impact on performance
        - **Setting method**: Concrete instructions on how to determine appropriate values
        - **Default value**: Recommended default with justification based on typical molecular systems

    ## Prohibited Approaches
    Note: Within the constraints, partial structures from the following prohibited circuits may be utilized if they contribute to hardware efficiency.

    - AdaptVQE: An iterative algorithm that adaptively grows the ansatz based on the gradient of operators with respect to the current state. This approach inherently requires multiple sequential optimization steps, which is incompatible with the single-pass parameter optimization constraint.

    - UCCSD and its Variants: Unitary Coupled Cluster Singles and Doubles, along with derivative formulations such as UCCGSD, are not permitted. These approaches typically introduce a prohibitively large number of variational parameters, leading to high classical optimization cost, potential barren plateaus, and poor scalability to larger systems.
    """
)

COST_FUNCTION_DEFINITION = dedent(
    """The cost function is defined as the mean absolute error between the Full Configuration Interaction (FCI) energy and the energy estimated by the Variational Quantum Eigensolver (VQE) across multiple molecules and bond lengths. This metric quantifies how accurately the VQE approach reproduces the ground state energy in comparison to the exact FCI results.

    The cost function is given by the following formula:
        Cost = \\sum_{i,j} |E_FCI^(i,j) - E_VQE^(i,j)| / (M * N)

    Where:
    - M is the number of molecules evaluated.
    - N is the number of bond lengths evaluated.
    - E_FCI^(i,j) is the Full CI ground state energy at the i-th molecule and j-th bond length.
    - E_VQE^(i,j) is the corresponding VQE estimated ground state energy at the i-th molecule and j-th bond length.
    """
)

MOLECULE_HAMILTONIAN_DEFINITION = dedent(
    """The Hamiltonian used in quantum chemistry calculations is given in second quantization form as:
    H = \\sum_{{p,q}} h_{{pq}} c†_p c_q + (1/2) \\sum_{{p,q,r,s}} h_{{pqrs}} c†_p c†_q c_r c_s

    This Hamiltonian describes the behavior of electrons in a molecular system. It is expressed in terms of fermionic creation (c†) and annihilation (c) operators, which satisfy anticommutation relations.
    - h_{{pq}} are the one-electron integrals, representing kinetic energy and nuclear attraction.
    - h_{{pqrs}} are the two-electron integrals, accounting for electron-electron Coulomb repulsion.
    - Indices p, q, r, s label spin orbitals.
    - The first term corresponds to one-body (single-electron) effects.
    - The second term represents two-body (electron-electron interaction) effects.
    """
)


OPTIMIZER_SETTINGS = dedent(
    """Use the SLSQP (Sequential Least Squares Quadratic Programming) optimizer to optimize the ansatz parameters. For each molecule, perform the optimization in order of increasing bond length. The optimized result from one bond length is used as the initial value for the next longer bond length. For the shortest bond length, the initial values are randomly assigned within the range [0.0, 0.1].
    """
)
