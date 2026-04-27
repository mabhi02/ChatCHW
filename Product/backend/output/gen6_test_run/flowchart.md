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

    subgraph SG_mod_initial_greeting["greeting the patient and confirming their identity (first)"]
        mod_initial_greeting_ENTRY(["Enter: greeting the patient and confirming the…"]):::module
    end

    subgraph SG_mod_emergency_escalation["immediately escalate to emergency protocol (first)"]
        mod_emergency_escalation_ENTRY(["Enter: immediately escalate to emergency proto…"]):::module
    end

    subgraph SG_mod_standard_assessment["proceed to standard assessment (first)"]
        mod_standard_assessment_ENTRY(["Enter: proceed to standard assessment"]):::module
    end

    subgraph SG_mod_flag_chart["flag the chart accordingly (first)"]
        mod_flag_chart_ENTRY(["Enter: flag the chart accordingly"]):::module
    end

    subgraph SG_mod_administer_analgesic["administer analgesic per standing order (first)"]
        mod_administer_analgesic_ENTRY(["Enter: administer analgesic per standing order"]):::module
    end

    subgraph SG_mod_continue_monitoring["continue monitoring (first)"]
        mod_continue_monitoring_ENTRY(["Enter: continue monitoring"]):::module
    end

    subgraph SG_mod_treatment_plan["treatment plan (first)"]
        mod_treatment_plan_ENTRY(["Enter: treatment plan"]):::module
    end

    subgraph SG_mod_surgical_consent_module["surgical consent module (first)"]
        mod_surgical_consent_module_ENTRY(["Enter: surgical consent module"]):::module
    end

    subgraph SG_mod_pharmacy_coordination_module["pharmacy coordination module (first)"]
        mod_pharmacy_coordination_module_ENTRY(["Enter: pharmacy coordination module"]):::module
    end

    subgraph SG_mod_geriatric_pathway["geriatric care coordination pathway (first)"]
        mod_geriatric_pathway_ENTRY(["Enter: geriatric care coordination pathway"]):::module
    end

    subgraph SG_mod_guardian_consent["ensure guardian consent is obtained (first)"]
        mod_guardian_consent_ENTRY(["Enter: ensure guardian consent is obtained"]):::module
    end

    subgraph SG_mod_patient_education["Explain the treatment options using language the patient can understand (first)"]
        mod_patient_education_ENTRY(["Enter: Explain the treatment options using lan…"]):::module
    end

    subgraph SG_mod_discharge_coordinator_role["discharge coordinator (first)"]
        mod_discharge_coordinator_role_ENTRY(["Enter: discharge coordinator"]):::module
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
    mod_initial_greeting_R0 --> INT_MERGE
    mod_emergency_escalation_R0 --> INT_MERGE
    mod_standard_assessment_R0 --> INT_MERGE
    mod_flag_chart_R0 --> INT_MERGE
    mod_administer_analgesic_R0 --> INT_MERGE
    mod_continue_monitoring_R0 --> INT_MERGE
    mod_treatment_plan_R0 --> INT_MERGE
    mod_surgical_consent_module_R0 --> INT_MERGE
    mod_pharmacy_coordination_module_R0 --> INT_MERGE
    mod_geriatric_pathway_R0 --> INT_MERGE
    mod_guardian_consent_R0 --> INT_MERGE
    mod_patient_education_R0 --> INT_MERGE
    mod_discharge_coordinator_role_R0 --> INT_MERGE
    INT_MERGE --> DONE