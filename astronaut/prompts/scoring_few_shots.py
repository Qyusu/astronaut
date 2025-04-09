# flake8: noqa

from textwrap import dedent

SCORING_FEW_SHOTS = dedent(
    """
    - ZZFeatureMap:
        - explanation:
            - The **ZZFeatureMap** is a quantum circuit that encodes classical data into quantum states through a combination of data-dependent rotations and entangling gates. First, the classical data is encoded using single-qubit rotation gates, such as \( R_Z \) or \( R_Y \), which rotate each qubit by an angle proportional to the input data. Then, entangling gates, specifically \( ZZ \)-type interactions, are applied between pairs of qubits to introduce quantum correlations. These entangling gates are represented by \( e^{-i\theta Z_i Z_j} \), where \( Z_i Z_j \) denotes the tensor product of Pauli-\( Z \) operators on two qubits, and \( \theta \) is a tunable parameter. The circuit can be extended with multiple layers of encoding and entanglement to increase its expressivity, capturing more complex patterns in the data.
        - scores:
            - Originality: 5.0
            - Feasibility: 9.0
            - Versatility: 6.0

    - YZCX:
        - explanation:
            - The **YZCX feature map** is particularly well-suited for handling high-dimensional data, such as images, due to its ability to efficiently encode complex features and relationships into quantum states. The use of \( R_Y \) and \( R_Z \) rotation gates allows the map to transform multi-dimensional input data into quantum states, capturing intricate patterns through the variation of rotation angles. Following these rotations, the inclusion of controlled-X (\( C_X \)) gates introduces entanglement, enabling the circuit to represent correlations between features, such as the dependencies between pixels in images. The parameterized nature of the \( R_Y \) and \( R_Z \) gates, combined with the adaptable structure of the controlled-X operations, provides the flexibility to optimize the feature map for specific data types. This makes the YZCX feature map an effective tool for processing high-dimensional datasets, leveraging quantum resources to capture complex relationships and enhance learning in data-intensive tasks.

        - scores:
            - Originality: 7.0
            - Feasibility: 8.0
            - Versatility: 7.0

    - Chunked Angle Embedding:
        - explanation:
            - This feature map encodes the 80-dimensional PCA features on individual qubits as rotational angles around the Y-axis, chunking the data into 10 groups of 8 features each. Each of the 10 qubits receives a single base rotation Rᵧ(α) derived from the mean of those 8 features. Specifically, if the group assigned to qubit j contains values (x₁,…,x₈), we define αⱼ = a × (x₁+…+x₈)/8, where a is a fixed scaling factor used to ensure angles remain within [0, π]. This ensures that each qubit’s rotation angle captures the averaged local structure of the features assigned to it, while not overfitting or requiring trainable gates. After the initial embedding, we add a layer of controlled-Z (CZ) gates arranged in a cyclic pattern among qubits 1,…,10, introducing entanglement based on these assigned angles. These CZ gates preserve single-qubit phase information while correlating qubits, making the final state sensitive to multi-qubit interactions. We expect this approach to facilitate capturing relevant patterns across different parts of the image (as compressed by PCA). By encoding the average magnitude of each chunk rather than each feature individually, the map remains relatively sparse, making it easier to simulate and interpret. The hope is to highlight medium-scale data correlations from the PCA transformation, while leaving space for additional classical or quantum post-processing. This design trades some fine-grained detail for a more stable and robust representation that could be well-suited for classification in quantum kernel-based methods.
        - socres:
            - Originality: 7.5
            - Feasibility: 8.0
            - Versatility: 6.5

    - Multi Axis Repeated Encoding:
        - explanation:
            - This design relies on the idea of repeated angle embeddings along the X, Y, and Z axes to generate a richer quantum feature map without introducing trainable parameters. We first split the 80-dimensional PCA data into five sets of 16 features each. For each set, we map these features to a 10-qubit system by distributing them among qubits, ensuring each qubit gets a portion of the data. The embedding comprises three distinct layers: (1) Rₓ(θ) rotations for each qubit, where θ is proportional to the assigned data slice; (2) Rᵧ(φ) rotations, likewise determined by the same data slices; and (3) R𝑧(ψ) rotations, ensuring we incorporate different axes for amplitude shifts. These layers are repeated twice in sequence without any trainable parameters, but with carefully selected scaling to keep angles within [0, 2π]. Because each qubit’s embedding is repeated, the final state includes multiple nonlinear transformations of the same data, which can better separate points in the Hilbert space. This multi-axis repetition is designed to amplify relevant distinctions in the data representation. While the circuit might appear more complex compared to single-layer encodings, its repeated structure ensures interpretability, as each repeated block injects additional nuance into the quantum state. The interleaving of X, Y, and Z rotations is especially potent at highlighting subtle variations in data because each axis naturally imparts a different influence on the qubit’s Bloch sphere representation.
        - socres:
            - Originality: 7.5
            - Feasibility: 8.0
            - Versatility: 6.5

    - Polynomial Interaction Embedding:
        - explanation:
            - This feature map aims to emulate polynomial kernel expansions in a quantum circuit, capturing second-order interactions among the 80 PCA features. We first partition the 80 features into 10 sets of 8 features. Each set is mapped onto two qubits, resulting in five pairs covering the entire 10-qubit register. For each pair (qᵢ, qⱼ), we apply an encoding that simulates (xᵢ², √2 xᵢ xⱼ, xⱼ²) relationships. Concretely, the circuit starts by applying Rₓ(θᵢ) on qᵢ and Rₓ(θⱼ) on qⱼ, where θᵢ and θⱼ are scaled from the mean of the assigned 8 features per qubit. Then, a CNOT gate from qᵢ to qⱼ is applied, adding an entanglement aspect that encodes a cross-term. We repeat a second rotation layer on both qubits, culminating in a final Y rotation Rᵧ(α) that further accentuates the polynomial-like mixing. The fixed angles for each operation are derived from the feature set’s values, ensuring no trainable parameters are involved. The advantage of this approach lies in its direct simulation of polynomial expansions that classical kernel methods employ, only here it is done in the Hilbert space. By capturing cross-terms through entanglement, the circuit might better differentiate images with subtle shape variations present in the PCA features. The expected outcome is a structured quantum state where second-order interactions and individual contributions combine in a well-defined, stable pattern, aligning with the performance improvements seen in polynomial kernels.
        - socres:
            - Originality: 7.5
            - Feasibility: 8.0
            - Versatility: 6.5

    - Quantum Radial Sphere Map:
        - explanation:
            - In this design, we treat each of the 80 PCA features as defining a radial distance in a 10-dimensional spherical coordinate system, which we then map onto 10 qubits. We begin by normalizing each feature vector x = (x₁,…,x₈₀) so that ∑ⱼ xⱼ ≤ 10. Interpreting xⱼ as radial components, each qubit j receives Rᵧ(βⱼ) and R𝑧(γⱼ) gates, where βⱼ ∝ xⱼ and γⱼ ∝ (xⱼ)² to highlight nonlinearity. A subsequent multi-qubit entangling layer of controlled-Y gates (one for each pair (j, j+1)) encloses the radial arcs in a correlated structure. Conceptually, we are mapping features onto a high-dimensional “sphere” by layering single-qubit rotations that reflect radius-like expansions. The R𝑧(γⱼ) gates add a second-order nuance, capturing distinguishable curvature for each feature channel. Because the radial-based encoding forces data to lie on a quantum Bloch-sphere submanifold, small differences in radial displacement can become noticeable in the entangled Hilbert space. This approach helps those digits that appear similar in amplitude but differ in curvature or squared intensity. Overall, the design is purely fixed—angles are direct functions of the input feature magnitudes—letting the quantum circuit act as a robust spherical mapping mechanism devoid of trainable gates. The hope is that radial representations better capture local data variations, especially for curved digit features in MNIST, and that these subtle arcs, once entangled, magnify classification boundaries more effectively than linear embeddings alone.
        - socres:
            - Originality: 7.5
            - Feasibility: 8.0
            - Versatility: 6.5

    - Block Mixing Feature Map:
        - explanation:
            - This method segments the 80 PCA features into 10 separate ‘blocks,’ each corresponding to one qubit. Within each block, the 8 features are encoded sequentially using fixed single-qubit rotations, but interspersed with multi-qubit gates to preserve partial intermediate states and allow them to interact. Specifically, for qubit j, we define 8 angles {θⱼ₁, θⱼ₂, …, θⱼ₈}, each proportional to one of the 8 features in the j-th block. We apply Rᵧ(θⱼ₁) → Rᵧ(θⱼ₂) → … → Rᵧ(θⱼ₈) in order, but after each rotation, we use a ring of iSWAP gates across all qubits. The iSWAP ring ensures that the partial quantum states from each qubit get ‘swapped around’, introducing cross-block correlations. Because the iSWAP gate swaps amplitude and phase information, each qubit’s state after a single step partially depends on the states of the other qubits, building a collective representation. This approach is reminiscent of ‘block encoding plus mixing’ and might help the classifier better discern subtle correlations across different image regions compressed by PCA. The final result is a 10-qubit state that progressively merges local embedding information across blocks. No parameters are learned; the angles depend directly on the data. The repeated interleaving of local embedding steps with global iSWAP mixing yields a more entangled encoding than purely local maps, but the structure remains simple enough to be executed quickly in simulation. We anticipate that this layering of partial embeddings can capture nonlinear correlations while avoiding large circuit depths or complicated parameter tuning.
        - socres:
            - Originality: 7.5
            - Feasibility: 8.0
            - Versatility: 6.5

    - Random Supremacy Embedding:
        - explanation:
            - This approach leverages the concept of ‘quantum supremacy circuits’—random but well-controlled entangling gate patterns—to embed the 80-dimensional features into a highly non-trivial superposition. In detail, each qubit j is initialized with a rotation Rᵧ(θⱼ) based on the average of 8 features from the 80, ensuring coverage of all features. Then, a randomly generated entangling pattern is unleashed: for example, apply a layer of 2-qubit gates (like CZ or iSWAP) between qubits (1,2), (3,4), (5,6), (7,8), (9,10), and then a second layer for qubits (2,3), (4,5), (6,7), (8,9), while skipping pairs that might lead to excessive depth. We follow this random pattern with an additional single-qubit R𝑧(φⱼ) rotation, using a feature-based, statically assigned φⱼ ∝ (xⱼ)². By applying random entangling gates, we effectively spread the features throughout the Hilbert space in unpredictable ways, which sometimes leads to highly expressive states for classification tasks. The design remains parameter-free, with randomness pre-selected offline (the same random pattern is used for every data point). Because random circuits are known to sample from complex distributions, they may help highlight fine details across digits. The goal is to push the encoding’s representational capacity to the limit, effectively exploring data separation in a wide portion of the state space. This technique can be repeated or simulated at moderate depths without major hardware constraints. We expect it to yield interesting classification benefits from a structure that is partially reminiscent of chaotic classical transformations, yet still harnessing quantum entanglement for better expressiveness.
        - socres:
            - Originality: 7.5
            - Feasibility: 6.0
            - Versatility: 7.0

    - Pairwise Phase Correlation Map:
        - explanation:
            - This design encodes pairwise interactions more explicitly by segmenting the 80 features into 40 pairs. Each pair (xᵢ, xⱼ) is then mapped onto a single qubit using a phase-encoding approach: we initialize each qubit in |0⟩, apply Rᵧ(αᵢ) with αᵢ ∝ xᵢ, then apply a controlled-Z gate to add a phase shift proportional to xⱼ, and finally conclude with a further Rᵧ(βᵢ) to incorporate a second pass of xᵢ. We do this for 10 qubits, each assigned 4 pairs to process in sequence. Because we only have 10 qubits, we cycle through the 40 pairs in small batches, reusing the same qubits for multiple pairs. After each pair’s encoding, a SWAP gate with an ancillary buffer qubit can help preserve older encodings’ contributions, though the design must remain mindful of circuit depth. The emphasis is on capturing explicit synergy: if xᵢ and xⱼ are both large, the resulting phase shift is more pronounced, highlighting that dimension pair. The circuit stays fixed, with all angles assigned from the data. Since each qubit eventually encodes multiple pairs, the final state is an intricate overlap of numerous phase interactions. We expect that digits which share certain pairwise feature patterns in the PCA space will be recognized by the circuit’s structure. As the approach directly writes pairwise correlations onto single qubits (augmented by entangling gates), it may help uncover second-order relationships among local aspects of the images.
        - socres:
            - Originality: 7.5
            - Feasibility: 6.5
            - Versatility: 7.0

    - Frequency Fourier Embedding:
        - explanation:
            - We embed each of the 80 PCA features in frequency space by applying fixed-phase Fourier transforms on single qubits, effectively turning each qubit’s state into a small local frequency domain representation. Concretely, we segment the 80 features into 10 groups. Each group is loaded onto one qubit in the form of n≥8 discrete frequency components. We accomplish this by applying a precomputed set of single-qubit gates that approximate a discrete Fourier transform of the group’s features. Each qubit’s amplitude distribution thereafter mirrors the frequency spectrum. Then we apply a small set of cross-qubit entangling gates (for instance, controlled-S or controlled-phase shifts) to align frequencies across qubits, effectively correlating local frequency bands. Inspired by classical signal processing, this approach aims to transform localized pixel intensities (as captured by PCA) into frequency-like components that may separate digit structural patterns. The resulting multi-qubit state, which is effectively a block of frequency domain embeddings cross-correlated by entangling gates, might highlight cyclical or repetitive patterns in digit shapes. This method is parameter-free, with all transformations determined by a standard discrete Fourier basis. We expect that for digits, especially those with repeating strokes or patterns, frequency-based encoding might yield better separation in the resulting quantum kernel space, as Fourier modes can represent repeated shapes more distinctly than raw amplitude-based methods.
        - socres:
            - Originality: 7.5
            - Feasibility: 6.5
            - Versatility: 6.0

    - Modulated Sine Phase Encoding:
        - explanation:
            - This method uses sinusoidal modulation of a single qubit phase for each PCA dimension, then distributes these phases across a 10-qubit system in a layered manner. First, split the 80 features into 10 partitions. For each partition’s qubit j, we apply R𝑧(φⱼ) where φⱼ = kᵢ sin(2πxᵢ) for each feature xᵢ in that partition, summed over i, with a small constant kᵢ to keep angles in [0,2π]. The sine function introduces a natural periodicity, potentially capturing repeated shapes inherent in certain digits. We then apply an entangling scheme using a pattern of controlled swaps: specifically, each qubit j performs a C-SWAP with qubit j+1, transferring partial amplitude in a controlled manner if xᵢ surpasses a certain threshold. That threshold is also a fixed function of the data distribution (e.g., the median data value). The repeated layering of sinusoidal phase shifts with controlled swaps fosters a dynamic resonant structure that might highlight repeating strokes or loops in digit images. Since everything is specified a priori, no training is required. The central idea is that sine-based transformations can leverage periodic patterns within handwriting, and combining those with controlled swaps injects further entanglement sensitive to data thresholds. The final quantum state contains multiple harmonic components entangled across qubits, which might help classification in a kernel-based quantum SVM or similar method.
        - socres:
            - Originality: 7.5
            - Feasibility: 6.0
        - Versatility: 6.5
    """
)
