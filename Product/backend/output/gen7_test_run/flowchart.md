graph TD
    classDef start fill:#6b7280,stroke:#374151,color:#fff
    classDef activator fill:#8b5cf6,stroke:#6d28d9,color:#fff
    classDef router fill:#0891b2,stroke:#0e7490,color:#fff
    classDef module fill:#4488ff,stroke:#2266cc,color:#fff
    classDef decision fill:#fef3c7,stroke:#92400e,color:#000
    classDef rule fill:#e0f2fe,stroke:#0369a1,color:#000
    classDef default_rule fill:#fef9c3,stroke:#ca8a04,color:#000
    classDef emergency fill:#dc2626,stroke:#7f1d1d,color:#fff,stroke-width:3px
    classDef integrative fill:#16a34a,stroke:#14532d,color:#fff
    classDef done fill:#059669,stroke:#064e3b,color:#fff

    START(["Patient arrives"]):::start

    subgraph SG_mod_danger_signs["General Danger Signs Assessment (first)"]
        mod_danger_signs_ENTRY(["Enter: General Danger Signs Assessment"]):::module
        mod_danger_signs_R0["(no outputs)"]:::default_rule
        mod_danger_signs_ENTRY -->|"default (else)"| mod_danger_signs_R0
        mod_danger_signs_R1["(no outputs)"]:::default_rule
        mod_danger_signs_ENTRY -->|"default (else)"| mod_danger_signs_R1
    end

    subgraph SG_mod_cough["Cough and Breathing Assessment (first)"]
        mod_cough_ENTRY(["Enter: Cough and Breathing Assessment"]):::module
        mod_cough_R0["(no outputs)"]:::default_rule
        mod_cough_ENTRY -->|"default (else)"| mod_cough_R0
        mod_cough_R1["(no outputs)"]:::default_rule
        mod_cough_ENTRY -->|"default (else)"| mod_cough_R1
        mod_cough_R2["(no outputs)"]:::default_rule
        mod_cough_ENTRY -->|"default (else)"| mod_cough_R2
        mod_cough_R3["(no outputs)"]:::default_rule
        mod_cough_ENTRY -->|"default (else)"| mod_cough_R3
    end

    subgraph SG_mod_fever["Fever and Malaria Assessment (first)"]
        mod_fever_ENTRY(["Enter: Fever and Malaria Assessment"]):::module
        mod_fever_R0["(no outputs)"]:::default_rule
        mod_fever_ENTRY -->|"default (else)"| mod_fever_R0
        mod_fever_R1["(no outputs)"]:::default_rule
        mod_fever_ENTRY -->|"default (else)"| mod_fever_R1
        mod_fever_R2["(no outputs)"]:::default_rule
        mod_fever_ENTRY -->|"default (else)"| mod_fever_R2
        mod_fever_R3["(no outputs)"]:::default_rule
        mod_fever_ENTRY -->|"default (else)"| mod_fever_R3
    end

    subgraph SG_mod_diarrhoea["Diarrhoea Assessment (first)"]
        mod_diarrhoea_ENTRY(["Enter: Diarrhoea Assessment"]):::module
        mod_diarrhoea_R0["(no outputs)"]:::default_rule
        mod_diarrhoea_ENTRY -->|"default (else)"| mod_diarrhoea_R0
        mod_diarrhoea_R1["(no outputs)"]:::default_rule
        mod_diarrhoea_ENTRY -->|"default (else)"| mod_diarrhoea_R1
        mod_diarrhoea_R2["(no outputs)"]:::default_rule
        mod_diarrhoea_ENTRY -->|"default (else)"| mod_diarrhoea_R2
        mod_diarrhoea_R3["(no outputs)"]:::default_rule
        mod_diarrhoea_ENTRY -->|"default (else)"| mod_diarrhoea_R3
    end

    subgraph SG_mod_malnutrition["Malnutrition Assessment (first)"]
        mod_malnutrition_ENTRY(["Enter: Malnutrition Assessment"]):::module
        mod_malnutrition_R0["(no outputs)"]:::default_rule
        mod_malnutrition_ENTRY -->|"default (else)"| mod_malnutrition_R0
        mod_malnutrition_R1["(no outputs)"]:::default_rule
        mod_malnutrition_ENTRY -->|"default (else)"| mod_malnutrition_R1
        mod_malnutrition_R2["(no outputs)"]:::default_rule
        mod_malnutrition_ENTRY -->|"default (else)"| mod_malnutrition_R2
    end

    subgraph INT["Integrative (merge outputs)"]
        INT_MERGE["Merge per-module outputs\n• highest referral wins\n• treatments additive (unless contraindicated)\n• shortest follow-up"]:::integrative
        INT_RULE_0["rule 0"]:::integrative
        INT_MERGE --> INT_RULE_0
        INT_RULE_1["rule 1"]:::integrative
        INT_MERGE --> INT_RULE_1
        INT_RULE_2["rule 2"]:::integrative
        INT_MERGE --> INT_RULE_2
        INT_RULE_3["rule 3"]:::integrative
        INT_MERGE --> INT_RULE_3
    end

    DONE(["Care plan complete"]):::done
    mod_danger_signs_R1 --> INT_MERGE
    mod_cough_R3 --> INT_MERGE
    mod_fever_R3 --> INT_MERGE
    mod_diarrhoea_R3 --> INT_MERGE
    mod_malnutrition_R2 --> INT_MERGE
    INT_MERGE --> DONE