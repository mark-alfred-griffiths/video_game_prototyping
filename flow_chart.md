# NPC Game Engine Architecture

```mermaid
flowchart LR

    %% =========================================================
    %% PLAYER + WORLD
    %% =========================================================

    A[Player Choice / Dialogue / Action]
    B[Observation<br/>strength<br/>reliability<br/>source]

    A --> B

    %% =========================================================
    %% FEATURE CONSTRUCTION
    %% =========================================================

    C[Raw Feature Construction<br/><br/>
    belief_mean<br/>
    alpha_scaled<br/>
    beta_scaled<br/>
    uncertainty<br/>
    confidence<br/>
    prior<br/>
    evidence_strength<br/>
    reliability<br/>
    weighted_strength<br/>
    trust<br/>
    suspicion<br/>
    instability]

    B --> C

    %% =========================================================
    %% PREDICTIVE CODING
    %% =========================================================

    D[JPC Predictive Coding Encoder<br/><br/>
    jpc.make_mlp(...)<br/>
    jpc.make_pc_step(...)<br/><br/>
    Learns latent representation z]

    C --> D

    %% =========================================================
    %% LATENT SPACE
    %% =========================================================

    E[Latent Representation z]

    D --> E

    %% =========================================================
    %% MLP HEADS
    %% =========================================================

    F[Belief Head - TensorFlow MLP<br/><br/>
    Outputs:<br/>
    raw_delta_alpha<br/>
    raw_delta_beta]

    G[Policy Head - TensorFlow MLP<br/><br/>
    Outputs:<br/>
    policy logits<br/><br/>
    dismiss / probe / reveal / confront]

    E --> F
    E --> G

    %% =========================================================
    %% BELIEF UPDATE
    %% =========================================================

    H[Belief Update<br/><br/>
    delta_alpha = softplus(raw_delta_alpha)<br/>
    delta_beta = softplus(raw_delta_beta)<br/><br/>
    alpha += delta_alpha<br/>
    beta += delta_beta]

    F --> H

    %% =========================================================
    %% ACTION SELECTION
    %% =========================================================

    I[Action Selection<br/><br/>
    argmax(policy_logits)<br/><br/>
    NPC Action]

    G --> I

    %% =========================================================
    %% NPC STATE UPDATE
    %% =========================================================

    J[NPC State Update<br/><br/>
    belief_mean<br/>
    uncertainty<br/>
    confidence<br/>
    memory<br/>
    trust<br/>
    suspicion<br/>
    instability]

    H --> J
    I --> J

    %% =========================================================
    %% CLOSED LOOP
    %% =========================================================

    J --> K[Next Game Step]
    K --> B

    %% =========================================================
    %% HARD-CODED FORWARD MODEL
    %% =========================================================

    subgraph FM[Hard-Coded Forward Model / Teacher Signals]

        FE[free_energy(...)]

        LT[pc_latent_teacher(...)<br/><br/>
        target latent z*]

        BT[belief_delta_teacher(...)<br/><br/>
        target delta_alpha<br/>
        target delta_beta]

        PT[policy_teacher(...)<br/><br/>
        target action label]

    end

    %% =========================================================
    %% TRAINING SIGNALS
    %% =========================================================

    LT -. training target .-> D
    BT -. belief supervision .-> F
    PT -. policy supervision .-> G
    FE -. free-energy signal .-> D

    %% =========================================================
    %% STYLING
    %% =========================================================

    classDef world fill:#f5f5f5,stroke:#333,stroke-width:1px;
    classDef pc fill:#dbeafe,stroke:#1e40af,stroke-width:2px;
    classDef latent fill:#ede9fe,stroke:#5b21b6,stroke-width:2px;
    classDef mlp fill:#dcfce7,stroke:#166534,stroke-width:2px;
    classDef update fill:#fef3c7,stroke:#92400e,stroke-width:2px;
    classDef forward fill:#fee2e2,stroke:#991b1b,stroke-width:2px;

    class A,B,C,J,K world;
    class D pc;
    class E latent;
    class F,G mlp;
    class H,I update;
    class FE,LT,BT,PT forward;
```

## Summary

This diagram shows the closed-loop NPC architecture:

```text
Player choice
→ observation
→ raw feature construction
→ JPC predictive-coding encoder
→ latent representation
→ TensorFlow MLP heads
→ belief update and action selection
→ updated NPC state
→ next game step
```

The hard-coded forward model supplies teacher signals for training the predictive-coding encoder and the MLP heads. The learned system then uses the encoded latent state to update Beta-distribution belief parameters and select an NPC action.
