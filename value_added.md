# Value added by CHW Navigator

This file restates the **Value added of CHW Navigator** stakeholder brief in Markdown (for version control and linking). A Word original, if used by your program, can live alongside this repo or in program archives; keep both in sync when the narrative changes.

---

## Executive summary

Community health workers (CHWs) deliver essential primary care but face **major structural challenges**. They often receive only **4–12 weeks of training**, yet are expected to follow complex **IMCI-style** protocols: accurate symptom elicitation, respiratory rate counting, chest indrawing, malaria RDTs, MUAC screening, and correct referral decisions. **Adherence to guidelines** in low- and middle-income settings is often **below 50%**, even among providers with more training than CHWs (Rowe et al. 2018).

Most deaths from **diarrhoea and pneumonia**—leading causes of under-five mortality—are **preventable** with early, correct treatment (ORS plus zinc, antimalarials when indicated, amoxicillin for pneumonia, etc.). Adherence stays low partly because **paper protocols** and **hand-coded digital tools** are hard to use, slow to update, and expensive to maintain.

The **CHW Clinical Copilot** (CHW Navigator) addresses this by generating **deterministic, offline, auditable** clinical workflows from **national or WHO guidelines**. **AI is used during development—not at the point of care**—to convert manuals into **decision tables (DMN)** and **flowcharts** clinicians can review, then into **deployment-ready forms** (e.g. XLSForm for CHT). The system can **surface gaps** (e.g. missing stock-out logic) **before** final coding.

**Safety** is supported by **transparent clinician review**, **multi-model validation**, and **synthetic-patient** testing across common and edge cases (danger signs, comorbidities, malnutrition, supplies). The **deployed app contains no generative AI**, reducing hallucination risk and aligning with expectations for **predictable, rule-based** systems.

**Localization** is simplified through a **phrase bank** that separates **clinical logic** from **language**, so clinicians can validate meaning and CHWs can validate clarity—aligned with WHO recommendations that many countries rarely achieve in full.

When **guidelines change**, ministries can review **only modified decision rows** and phrases; **automated testing** helps confirm that behavior changes **only where the guideline changed**.

**In short:** CHW Navigator aims to help governments deploy **accurate, up-to-date, evidence-based** guidance to every CHW **offline and at scale**, improving diagnosis and treatment for leading causes of child death.

---

## 1. CHW challenges without a digital assistant

### 1.1 Structural limits

CHWs have **short training**; IMCI-style guidelines require **dozens of steps**. Complexity drives low adherence.

- CHWs often **miss danger signs** or **misdiagnose** (Downey et al. 2021; Lal et al. 2016; WHO 2024).
- In many LMICs, **frontline adherence to guidelines** is **under half the time** (Rowe et al. 2018; WHO reviews).

### 1.2 Where paper protocols break

- **Missing supplies** (no ORS, broken scale, no MUAC): manuals often lack **fallback** logic, so decisions stall.
- **Referral criteria** interact across danger signs; CHWs struggle to stay **consistent**.

### 1.3 Weak validation of paper workflows

- Manuals are **long** (200+ pages); subtle inconsistencies are hard for MOH staff to catch.
- There is usually **no formal test** that “tomorrow’s advice” matches “today’s” for the same case presentation.

---

## 2. Problems with hand-coded digital modules

CHW platforms (CHT, OpenSRP, CommCare, etc.) are powerful but **costly** to build and maintain.

### 2.1 Slow and expensive

- Each chapter (cough vs diarrhoea, etc.) needs **interpretation**, **flow logic**, and often **custom code per platform**.
- Programmers must learn **CHW-specific** patterns (timers, visits, referrals).
- Many programs only ship **1–3 conditions** because full digitization is unaffordable.

### 2.2 Uncaptured ambiguities in manuals

- Gaps: **broken equipment**, **stock-outs**, **contradictory referral rules**, **unclear age thresholds**.
- These often surface **late**, after code is written.

### 2.3 Ad hoc clinical validation

- Reviewers **click through** screens; they cannot exhaust **combinations** (e.g. danger signs + malnutrition + comorbidity).
- Typically **no automated synthetic-patient suite**.

### 2.4 Painful updates

- New recommendations (e.g. **WHO 2024** respiratory updates, dosing changes) force **rewrites across platforms**.

### 2.5 Brittle translation

- Common pattern: English strings from clinicians or programmers; **one-off translation** in capital cities.
- **CHW-reviewed** translations are rare; **text, logic, and updates** are not linked—**inconsistency risk**.

---

## 3. Value proposition: an auditable pipeline

If CHWs have a **simple, reliable** assistant at the point of care, they can **diagnose and treat** more accurately. Evidence links **early, appropriate** diarrhoea and pneumonia care to **large reductions in preventable mortality** (Rowe et al. 2018; WHO 2011; WHO 2024).

The approach uses an **AI-assisted build pipeline** to turn a CHW **manual or guideline** into **static, verifiable** artifacts. **Deployment** uses **standard formats** (e.g. XLSForm for CHT). **In the field**, forms can run **offline** on typical Android phones.

This aligns with **WHO AI guidance (2023)**, **computable guidelines** and **DAK / SMART Guidelines** directions (2022), CHW-friendly decision support, and digital adaptation kits for IMCI.

**Goal (quantitative ambition in the source doc):** cut the **cost and time** of a **new module or major update** by **at least 80%** while **improving quality**—to be validated with partner evidence.

### Value proposition 1: Faster creation with less “telephone game”

**(a) Automated logic extraction** — Ingest the PDF and produce:

- **DMN-style decision tables** clinicians can review (industry-standard DMN).
- **Visual flowcharts** for audit.
- **Deployment-oriented outputs**: XLSForm (CHT), XML (CommCare), FHIR Questionnaire (OpenSRP), etc.

**Result:** The system **automates translation from the manual**; it does **not replace clinicians** or invent clinical knowledge—addressing concerns in WHO AI-for-health guidance.

**(b) Proactive gap analysis** — Before final code generation, flag items for medical review, for example:

- Missing logic when **equipment or supplies** are unavailable (e.g. ORS stock-out).
- **Inconsistent thresholds**, circular or unreachable branches.
- **Low-quality scans** or ambiguous parsing.

This pushes **clarification upstream** before programming.

### Value proposition 2: Trusted clinical safety

**(a) Transparent artifacts** — Clinicians review **flowcharts**, **decision tables**, and **deltas vs prior versions**—easier and safer than reviewing raw XLSForm XML or hand-written JS.

**(b) Multi-model validation** — Multiple LLMs plus **static checks** for age thresholds, referral ordering, underspecified actions, stock-out fallbacks, etc.

**(c) Automated stress testing** — Synthetic patients covering danger-sign combinations, **threshold edges**, supply permutations, and **comorbidities**.

**(d) Deterministic runtime** — Unlike chatbots, **deployed logic** is **deterministic** (e.g. XLSForm calculates)—**no generative model at the bedside**.

### Value proposition 3: Deployment and adoption

**(a) Native integration** — Outputs use **familiar form patterns** and standard platform hooks (e.g. follow-up, timers, referrals, stock events, registry lookups—exact APIs depend on the platform).

**(b) Familiar UX** — CHWs see **forms like those they already use** (where the program adopts that integration pattern).

**(c) Offline-first** — Compiles into **offline-first** Android experiences.

**(d) Phrase-bank localization** — Logic and language **decouple**; translation can respect grammar and register; **clinicians** review clinical meaning and **CHWs** review clarity.

### Value proposition 4: Maintenance and updates

**(a) Delta-based review** — On manual change: highlight **changed decision rows** and **changed phrases** so doctors review a **small fraction** of the full manual, with an **audit trail**.

**(b) Regression-style testing** — For synthetic cases: compare **old vs new** advice, show **where treatment changed**, and check that **unchanged guidelines** imply **unchanged outputs** for fixed scenarios—important for **trust**.

---

## 4. Before / after summary

| Feature | Status quo | CHW Navigator / Copilot |
|--------|------------|-------------------------|
| Logic creation | Clinician → programmer “telephone game” | Pipeline extracts logic from the manual (with human gates) |
| Gaps in guidelines | Found late | Gaps flagged **before** coding |
| Clinical review | Many screens / scenarios clicked manually | Doctors review **flowcharts and/or decision tables** |
| Safety testing | None or manual | **Synthetic-patient** and automated checks (aspiration / roadmap) |
| Updates | Full rebuild | **Delta** review + tests showing change only where policy changed |
| Translations | One-off, often capital-only | **Phrase bank** with clinician + CHW review |
| Offline | Sometimes limited | **Fully offline** deterministic runtime |

---

## Engineering and product roadmap (from source doc)

These bullets summarize the **“Engineering introduction”** section of the source brief; implementation status varies—see `Product/ARCHITECTURE.md`, `Product/PIPELINE.md`, and **`workflow.md`** at the repository root.

- **Needs assessment:** interviews with NGOs and MOHs; validate pain points and product fit.
- **Core pipeline:** quality-check manuals; clinician feedback for rough edges; decompose manual into facts; assemble **workflow + DMN**; DMN → flowcharts; clinician sign-off; DMN → **XLSForm** (CHT first), then **FHIR Questionnaire** (OpenSRP, ABDM / ASHA contexts, etc.); platform-specific integrations (follow-up, SMS, etc.).
- **Quality at each step:** red-team LLM passes, linters, unit and integration tests; **end-to-end** runs with **many synthetic patients**; phrase bank creation and translation with **clinician + CHW** review; **orchestrator + web UI** for medical directors.
- **Field testing:** partner NGO or district; qualitative research; randomized trials on **adherence with simulated patients** (e.g. patient-bot); later phases—RCTs with real patients under MD oversight.

---

## Field testing (aspirational)

- Partner with an **NGO or district**.
- **Qualitative** research with managers and CHWs.
- **Randomized trial**: CHW adherence on **simulated** cases with vs without the tool (may use LLM-assisted **patient bot**).
- Later: **RCT** with real patients and MD oversight; broader field deployment.

---

## References

- Agarwal, S., et al. (2016). *Tropical Medicine & International Health*, 20(8), 1003–1014.
- Downey, J., et al. (2021). *PLoS ONE* 16(3): e0246773. https://doi.org/10.1371/journal.pone.0246773  
- Lal, S., et al. (2016). *Malaria Journal* 15: 216. https://doi.org/10.1186/s12936-016-1205-6  
- Rowe, A. K., et al. (2018). *The Lancet Global Health*, 6(10), e1063–e1081. (Aggregates LMIC contexts.)
- WHO (2011). *Caring for the Sick Child in the Community.* Geneva: WHO.
- WHO (2022). SMART Guidelines / Digital Adaptation Kits. https://www.who.int/publications/m/item/who-digital-accelerator-kits  
- WHO (2023). *Regulatory considerations on artificial intelligence for health.* Geneva: WHO.
- WHO (2024). Guideline on management of infants and children with acute respiratory infection. Geneva: WHO.

---

## Appendix: CHW Navigator vs AI at the point of care

| Dimension | AI advising at point of care | CHW Navigator / Copilot |
|-----------|-------------------------------|-------------------------|
| Philosophy | Model interprets manual live | Builds the **same kind of forms** platforms already use, from the manual |
| Hardware | Often needs strong devices or connectivity | **Offline** on typical Android phones |
| User interface | Flexible but often **novel** | **Familiar** forms where integrated that way |
| Predictability | Hallucination and drift risk | **Deterministic** at point of care; validated **before** deployment |
| Regulation | Often complex | Aims to avoid **new** regulatory class by staying rule-based at bedside |

---

*Update this Markdown when the stakeholder Word/PDF brief changes materially.*
