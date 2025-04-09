# flake8: noqa

from textwrap import dedent

QUANTUM_GATE_LIST = dedent(
    """
    - **CNOT**: Controlled-NOT gate
        - Controlled-NOT gate (CNOT) is a two-qubit gate that flips the second qubit if the first qubit is in the state |1>.

    - **ControlledPhaseShift**: Controlled-phase shift gate  
        - Applies a phase shift to the target qubit if the control qubit is in the state |1>.

    - **CRot**: Controlled arbitrary rotation gate  
        - Applies an arbitrary rotation to the target qubit around a specified axis, conditioned on the control qubit being in the state |1>.

    - **CRX**: Controlled Rotation X gate  
        - Rotates the target qubit about the X axis by a specified angle, conditioned on the control qubit being in the state |1>.

    - **CRY**: Controlled Rotation Y gate  
        - Rotates the target qubit about the Y axis by a specified angle, conditioned on the control qubit being in the state |1>.

    - **CRZ**: Controlled Rotation Z gate  
        - Rotates the target qubit about the Z axis by a specified angle, conditioned on the control qubit being in the state |1>.

    - **CSWAP**: Controlled Swap gate  
        - Swaps the states of two target qubits if the control qubit is in the state |1>.

    - **CY**: Controlled Pauli Y gate  
        - Applies the Pauli Y gate to the target qubit if the control qubit is in the state |1>.

    - **CZ**: Controlled Pauli Z gate  
        - Applies the Pauli Z gate to the target qubit if the control qubit is in the state |1>.

    - **Hadamard**: Hadamard gate  
        - Creates a superposition by applying a 50/50 probability of measuring the qubit in state |0> or |1>.

    - **IssingXX**: Ising XX coupling gate  
        - Applies an XX interaction between two qubits with a specified coupling strength.

    - **IssingXY**: Ising (XX + YY) coupling gate  
        - Applies an XX + YY interaction between two qubits with a specified coupling strength.

    - **IssingYY**: Ising YY coupling gate  
        - Applies a YY interaction between two qubits with a specified coupling strength.

    - **IssingZZ**: Ising ZZ coupling gate  
        - Applies a ZZ interaction between two qubits with a specified coupling strength.

    - **ISWAP**: The i-SWAP gate  
        - Swaps the states of two qubits and introduces a phase of i (imaginary unit) to the swapped states.

    - **MultiControlledX**: Multi-controlled Pauli X gate  
        - Applies the Pauli X gate to a target qubit if all control qubits are in the state |1>.

    - **MultiRZ**: Arbitrary multi Z rotation gate  
        - Rotates multiple qubits collectively about the Z axis by a specified angle.

    - **OrbitalRotation**: Spin-adapted spatial orbital rotation gate  
        - Simulates spin-adapted orbital rotations in quantum chemistry calculations.

    - **PauliX**: Pauli X gate  
        - Flips the state of a qubit, equivalent to a NOT operation.

    - **PauliY**: Pauli Y gate  
        - Applies a 180-degree rotation about the Y axis to a qubit.

    - **PauliZ**: Pauli Z gate  
        - Applies a phase flip to the state |1> of a qubit.

    - **PhaseShift**: Arbitrary single-qubit local phase shift gate  
        - Introduces a phase shift to the state of a single qubit.

    - **PSWAP**: The phase-SWAP gate  
        - Swaps the states of two qubits with an additional phase applied to the swapped states.

    - **Rot**: Arbitrary single qubit rotation gate  
        - Rotates a single qubit about the X, Y, and Z axes by specified angles.

    - **RX**: Single-qubit rotation about the X axis gate  
        - Rotates a single qubit about the X axis by a specified angle.

    - **RY**: Single-qubit rotation about the Y axis gate  
        - Rotates a single qubit about the Y axis by a specified angle.

    - **RZ**: Single-qubit rotation about the Z axis gate  
        - Rotates a single qubit about the Z axis by a specified angle.

    - **S**: The single-qubit phase gate  
        - Introduces a phase shift of π/2 to the state |1> of a qubit.

    - **SWAP**: The SWAP gate  
        - Exchanges the states of two qubits.

    - **T**: The single-qubit T gate  
        - Introduces a phase shift of π/4 to the state |1> of a qubit.

    - **Toffoli**: The Toffoli (CCNOT) gate  
        - Flips the state of a target qubit if two control qubits are in the state |1>.
    """
)
