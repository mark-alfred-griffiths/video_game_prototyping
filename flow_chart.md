````mermaid
%%{init: {
  "theme": "base",
  "themeVariables": {
    "background": "#ffffff",
    "mainBkg": "#ffffff",
    "primaryTextColor": "#000000",
    "secondaryTextColor": "#000000",
    "tertiaryTextColor": "#000000",
    "lineColor": "#000000",
    "fontFamily": "Arial",
    "fontSize": "18px"
  }
}}%%

flowchart TD

    A["<b>Player Choice</b><br/><b>dialogue / action</b>"]

    B["<b>Observation</b><br/><b>strength · reliability · source</b>"]

    C["<b>Raw Feature Vector</b><br/><br/>
    <b>belief</b><br/>
    <b>uncertainty</b><br/>
    <b>prior</b><br/>
    <b>evidence</b><br/>
    <b>trust</b><br/>
    <b>suspicion</b><br/>
    <b>instability</b>"]

    A --> B
    B --> C

    %% =====================================================
    %% HARD-CODED FORWARD MODEL
    %% =====================================================

    subgraph M["<b>Hard-coded forward model used during training</b>"]
        direction LR

        M2["<b>latent teacher</b>"]

        M1["<b>free energy</b>"]

        M3["<b>belief delta teacher</b>"]

        M4["<b>policy teacher</b>"]
    end

    C -->|"input state"| M

    %% =====================================================
    %% PREDICTIVE CODING
    %% =====================================================

    D["<b>JPC Predictive Coding Encoder</b><br/><br/>
    <b>raw features → latent z</b>"]

    E["<b>Latent Representation</b><br/><b>z</b>"]

    M2 -->|"trains"| D

    M1 -.->|"shapes prediction error"| D

    D --> E

    %% =====================================================
    %% TENSORFLOW MLP HEADS
    %% =====================================================

    F["<b>Belief Head</b><br/>
    <b>TensorFlow MLP</b><br/><br/>
    <b>predicts delta alpha</b><br/>
    <b>predicts delta beta</b>"]

    G["<b>Policy Head</b><br/>
    <b>TensorFlow MLP</b><br/><br/>
    <b>predicts action logits</b>"]

    E --> F
    E --> G

    M3 -.->|"supervises"| F

    M4 -.->|"supervises"| G

    %% =====================================================
    %% BELIEF UPDATE + ACTION SELECTION
    %% =====================================================

    H["<b>Belief Update</b><br/><br/>
    <b>alpha += softplus</b><br/>
    <b>delta alpha</b><br/><br/>
    <b>beta += softplus</b><br/>
    <b>delta beta</b>"]

    I["<b>Action Selection</b><br/><br/>
    <b>dismiss</b><br/>
    <b>probe</b><br/>
    <b>reveal</b><br/>
    <b>confront</b>"]

    F --> H
    G --> I

    %% =====================================================
    %% NPC STATE UPDATE
    %% =====================================================

    J["<b>NPC State Update</b><br/><br/>
    <b>belief</b><br/>
    <b>uncertainty</b><br/>
    <b>confidence</b><br/>
    <b>memory</b><br/>
    <b>trust</b><br/>
    <b>suspicion</b><br/>
    <b>instability</b>"]

    H --> J
    I --> J

    %% =====================================================
    %% CLOSED LOOP
    %% =====================================================

    K["<b>Next Game Step</b>"]

    J --> K
    K --> A

    %% =====================================================
    %% STYLING
    %% =====================================================

    classDef default fill:#ffffff,color:#000000,stroke:#000000,stroke-width:2px;

    classDef input fill:#ffffff,color:#000000,stroke:#000000,stroke-width:2px;

    classDef model fill:#fde2e2,color:#000000,stroke:#dc2626,stroke-width:3px;

    classDef teacher fill:#fff1f1,color:#000000,stroke:#dc2626,stroke-width:2px;

    classDef pc fill:#dbeafe,color:#000000,stroke:#2563eb,stroke-width:3px;

    classDef latent fill:#f3e8ff,color:#000000,stroke:#7e22ce,stroke-width:3px;

    classDef mlp fill:#dcfce7,color:#000000,stroke:#16a34a,stroke-width:3px;

    classDef update fill:#fef3c7,color:#000000,stroke:#d97706,stroke-width:3px;

    class A,B,C,J,K input;

    class M model;

    class M1,M2,M3,M4 teacher;

    class D pc;

    class E latent;

    class F,G mlp;

    class H,I update;

    linkStyle default stroke:#000000,stroke-width:2px;
```
