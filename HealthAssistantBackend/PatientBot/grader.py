import json
import re
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

@dataclass
class GradingCriteria:
    """Represents grading criteria for a specific diagnosis"""
    diagnosis: str
    required_questions: List[str]
    optional_questions: List[str] 
    required_exams: List[str]
    final_diagnosis: str
    treatment_plan: str
    followup_instructions: str

class MedicalBotGrader:
    def __init__(self):
        self.grading_rubric = self._initialize_grading_rubric()
    
    def _initialize_grading_rubric(self) -> Dict[str, GradingCriteria]:
        """Initialize the complete grading rubric based on the medical guidelines"""
        return {
            "cough_chest_indrawing": GradingCriteria(
                diagnosis="Cough and difficulty breathing (e.g. child has chest indrawing)",
                required_questions=[
                    "Ask: For how long has the child had the cough? (duration helps identify chronic cough ≥14 days)",
                    "Ask about danger signs: Is the child unable to drink or breastfeed?", 
                    "Has the child had convulsions during this illness?"
                ],
                optional_questions=[
                    "Ask if the child has had fever in the last 3 days (to check for co-infection like malaria)",
                    "Ask the caregiver if there are any other concerns or symptoms not already mentioned (open-ended, to not miss other problems)"
                ],
                required_exams=[
                    "Look for chest indrawing in the lower chest wall when the child breathes in (sign of severe pneumonia)",
                    "Count the respiratory rate for 1 full minute to detect fast breathing (≥50 breaths/min if 2–12 mo; ≥40 if 12 mo–5 yr)",
                    "Check for malnutrition: measure MUAC and check for edema (to identify severe malnutrition)"
                ],
                final_diagnosis="Severe pneumonia (cough with chest indrawing present – a danger sign)",
                treatment_plan="Urgent referral to health facility is required for chest indrawing (severe pneumonia). Begin pre-referral treatment: give first dose of oral antibiotic (amoxicillin) if the child can swallow before sending to hospital. Ensure the child is kept warm and reassured during transfer.",
                followup_instructions="Immediate referral – advise caregiver to go to the hospital without delay. Provide a referral note and assist with transport if needed. No routine CHW follow-up (child will be managed at facility), but the CHW should check on the family soon after to confirm the child received care and to provide support after hospital treatment."
            ),
            "simple_cough": GradingCriteria(
                diagnosis="Cough or cold (no fast breathing, no danger signs)",
                required_questions=[
                    "Ask: For how long has the child had the cough? (to ensure it's <14 days)",
                    "Ask if difficulty feeding or drinking (to rule out danger sign)"
                ],
                optional_questions=[
                    "Ask if the child's nose is congested or if any home remedies were tried for the cough (helps in advising care, but doesn't change diagnosis)",
                    "Confirm no fever or other complaints (to ensure it's just a simple cough)"
                ],
                required_exams=[
                    "Count breaths to confirm breathing rate is normal (no fast breathing for age)",
                    "Check for chest indrawing (none, in this scenario)",
                    "Check MUAC/edema to assess nutritional status (routine check)"
                ],
                final_diagnosis="No pneumonia (Simple cough/cold) – cough <14 days, normal breathing, no danger signs",
                treatment_plan="No antibiotics needed (explain that without fast breathing, it's likely a mild cough/cold). Advise home care: use a safe soothing remedy like a bit of honey in warm water to calm the cough (for child >6 months). Ensure the child continues feeding well and fluids. Warn against cough syrups or unsafe medicines.",
                followup_instructions="Counsel the caregiver to monitor the child at home. No scheduled follow-up is required for a simple cold, but advise to return if the cough lasts ≥14 days, or if breathing becomes fast or difficult, or any new danger sign develops."
            ),
            "pneumonia_fast_breathing": GradingCriteria(
                diagnosis="Cough with fast breathing (no chest indrawing, no other danger sign)",
                required_questions=[
                    "Ask: How long has the child been coughing? (to rule out chronic cough ≥14 days)",
                    "Ask if fever is present or any vomiting (fever may indicate co-infection; vomiting could affect oral treatment)"
                ],
                optional_questions=[
                    "Ask if the child is wheezing or has asthma history (optional, may not alter initial pneumonia treatment but useful context)",
                    "Inquire about previous treatment given for the cough (to know if the caregiver started any medicines)"
                ],
                required_exams=[
                    "Count respiratory rate to confirm fast breathing above age threshold (sign of pneumonia)",
                    "Check for chest indrawing (none in this case, distinguishes from severe pneumonia)",
                    "Check temperature (fever) and perform a malaria RDT if fever is present (to detect malaria)",
                    "Assess nutrition (MUAC strap color, edema) as part of full exam"
                ],
                final_diagnosis="Pneumonia (cough with fast breathing, but no chest indrawing or other danger sign)",
                treatment_plan="Oral antibiotic: start a course of amoxicillin for pneumonia (appropriate dose for age, 2x daily for 5 days). Have the caregiver give the first dose under observation now. If the child also tested positive for malaria, begin antimalarial treatment as well (treat both illnesses) – e.g. give first dose of ACT now. Advise continued breastfeeding and fluids.",
                followup_instructions="Follow up in 3 days to assess recovery. At the follow-up, check breathing rate and overall condition. If child is not improving (still fast breathing or new fever) or gets worse before then, refer to a health facility. Advise caregiver to return sooner if any danger sign develops."
            ),
            "chronic_cough": GradingCriteria(
                diagnosis="Cough ≥14 days (chronic cough)",
                required_questions=[
                    "Ask in detail: Exactly how many days has the cough been present? (confirm it is two weeks or more)",
                    "Ask about weight loss or contact with someone with tuberculosis (to gather possible cause, though management is referral)"
                ],
                optional_questions=[
                    "Ask if the cough is improving at all or steadily worsening (for additional context)",
                    "Ask if caregiver sought any prior treatment over these two weeks (optional background)"
                ],
                required_exams=[
                    "Assess breathing: count rate and check for any fast breathing or chest indrawing. (Even if present, a ≥14 day cough is a danger sign requiring referral)",
                    "Check for fever or other signs (to note if multiple issues)",
                    "Nutritional assessment (MUAC/edema), as prolonged illness may cause malnutrition"
                ],
                final_diagnosis="Persistent cough (cough ≥14 days – possible TB, asthma, or pertussis) – a danger sign requiring further evaluation",
                treatment_plan="Refer to health facility for assessment of chronic cough (possible tuberculosis or other chronic illness). No community treatment is given for the cough's cause before referral (provide symptomatic relief only). If the child has fast breathing, you may give the first dose of amoxicillin to treat possible pneumonia on the way, but primary management is referral for investigations.",
                followup_instructions="Urgent referral (same day). Advise the caregiver that the child needs tests/assessment at the clinic for a long-lasting cough. Instruct to keep the child away from other young children if TB is suspected (until evaluated). After referral, coordinate with the clinic on follow-up. The CHW should follow up on the outcome within a week."
            ),
            "diarrhea_no_dehydration": GradingCriteria(
                diagnosis="Diarrhea (no dehydration) (loose stools <14 days, child active)",
                required_questions=[
                    "Ask: For how long has the child had diarrhea (loose stools)? (distinguish acute vs. ≥14 days)",
                    "Ask: Is there blood in the stool? (to check for dysentery)"
                ],
                optional_questions=[
                    "Ask about the child's fluid intake and urine output (optional – helps gauge hydration)",
                    "Ask if any home fluids or remedies have been given for the diarrhea (to reinforce good practices or correct any wrong ones)"
                ],
                required_exams=[
                    "Assess dehydration signs: check if eyes look sunken, skin pinch elasticity (if trained), and observe if child is thirsty. In this case, no notable dehydration signs (child is alert, not thirsty)",
                    "Check for fever (could indicate concurrent infection)",
                    "Weigh the child or check MUAC, as diarrhea can quickly affect nutrition (ensure MUAC is not in red/yellow indicating malnutrition)"
                ],
                final_diagnosis="Uncomplicated diarrhea (acute diarrhea <14 days, no blood, no signs of dehydration)",
                treatment_plan="Oral rehydration therapy (ORT): Give fluids and oral rehydration salts. Demonstrate ORS preparation and help the caregiver give the child ORS solution on the spot until the child is no longer thirsty. Send the caregiver home with at least 2 packets of ORS and instructions to give as much as the child wants, at least ½ cup after each loose stool. Zinc supplementation: Give zinc (age-appropriate dose: e.g. 1/2 tablet daily for <6 mo, 1 tablet for older, for 10 days) to reduce severity and prevent future episodes. Ensure the first zinc dose is given in your presence. Feeding advice: Continue regular breastfeeding and feeding through the illness (do not withhold food) and give extra fluid (clean water, soups) at home.",
                followup_instructions="Follow up in 3 days to check the child's status. At follow-up, assess for dehydration, weigh if possible, and ask if diarrhea has stopped. If not improving or if diarrhea has persisted to 14 days, refer to a health facility. Advise caregiver to return earlier if the child develops blood in stool, cannot drink, or becomes worse."
            ),
            "diarrhea_dehydration": GradingCriteria(
                diagnosis="Diarrhea with some dehydration (child is drinking eagerly, thirsty)",
                required_questions=[
                    "Ask: How many days of diarrhea? (to confirm it's acute <14 days)",
                    "Ask: Is there blood in the stool? (to rule out dysentery)",
                    "Ask if the child is able to drink/breastfeed or is vomiting everything (important for hydration status)"
                ],
                optional_questions=[
                    "Ask if caregiver noticed sunken eyes or decreased urine (additional dehydration signs)",
                    "Ask what fluids the caregiver has been giving at home (optional, to encourage continued ORS use)"
                ],
                required_exams=[
                    "Assess dehydration: the child is thirsty, drinks eagerly. Eyes may appear slightly sunken and mouth dry – these indicate some dehydration (but child is still able to drink)",
                    "Ensure no general danger sign: the child is not lethargic/unconscious and can drink (so not severe dehydration)",
                    "Check MUAC/edema to see if there's underlying malnutrition aggravating illness"
                ],
                final_diagnosis="Acute diarrhea with some dehydration (child is moderately dehydrated but no danger signs)",
                treatment_plan="Rehydrate immediately: Begin supervised ORS therapy. Have the caregiver slowly spoon-feed ORS, aiming for e.g. ~75 ml/kg over 4 hours (or 'drink until thirst is quenched'). Monitor the child's condition during this period. Once the child is rehydrated (no longer thirsty, more alert), send ORS packets home (at least 2) with instructions to continue ORS after every loose stool. Zinc supplementation for 10 days (age-appropriate dose). Give first dose now. Advise caregiver to continue feeding and breastfeeding frequently, even during rehydration.",
                followup_instructions="Follow up in 3 days to ensure hydration has been maintained and diarrhea is improving. At follow-up, reassess dehydration status and ask if diarrhea frequency has decreased. If not improving or if any danger signs (e.g. unable to drink) developed, refer the child. Emphasize caregiver should return immediately if child gets worse (e.g. persistent vomiting or refusal to drink)."
            ),
            "diarrhea_blood": GradingCriteria(
                diagnosis="Diarrhea with blood in stool (dysentery)",
                required_questions=[
                    "Ask: How long has the child had diarrhea? (often dysentery is acute, but check if ≥14 days)",
                    "Ask: Do you see blood in the stool? (caregiver's report of red blood or dark sticky stool)"
                ],
                optional_questions=[
                    "Ask about fever or abdominal pain (common in dysentery, though management still the same referral)",
                    "Ask if any antibiotics were already given at home (to inform the health facility)"
                ],
                required_exams=[
                    "Confirm blood in stool: if possible, inspect the diaper or ask caregiver to describe stool appearance (presence of blood and mucus indicates dysentery)",
                    "Check for dehydration and general condition (dysentery can also cause dehydration)",
                    "Measure MUAC to identify malnutrition, which could worsen outcomes"
                ],
                final_diagnosis="Dysentery (diarrhea with blood in stool) – serious bacterial infection (likely Shigella) that cannot be treated with the CHW's kit",
                treatment_plan="Refer to health facility for appropriate antibiotics. Explain that blood in stool (dysentery) needs medicine (antibiotic) that the CHW does not carry. Begin ORS to prevent dehydration while arranging referral (give ORS as tolerated). Do not delay referral – start ORS but prioritize getting child to clinic for antibiotic therapy. Advise caregiver to continue breastfeeding on the way and keep the child warm.",
                followup_instructions="Same-day referral. Instruct caregiver to go to the clinic or hospital immediately for treatment. Provide a referral note indicating 'bloody diarrhea'. After treatment, the CHW should follow up in ~3 days to ensure the child is recovering. If unable to reach a facility immediately, the caregiver should continue ORS and return to the CHW or facility the next day at the latest."
            ),
            "diarrhea_persistent": GradingCriteria(
                diagnosis="Diarrhea ≥14 days (persistent diarrhea)",
                required_questions=[
                    "Ask: Has the child had diarrhea for two weeks or more? (confirm the duration)",
                    "Ask about weight loss over this period and what the child is being fed (persistent diarrhea often causes malnutrition)"
                ],
                optional_questions=[
                    "Ask if stools are sometimes improving or continuously loose each day (to distinguish persistent vs. recurring)",
                    "Ask if any treatment was attempted in those 14+ days (to inform referral)"
                ],
                required_exams=[
                    "Assess hydration and weight: Even if hydrated now, prolonged diarrhea may cause malnutrition – check MUAC and weight trend if available",
                    "Check for any blood in stool (could be chronic dysentery) and other signs like thrush or persistent cough (to note if underlying condition like HIV, but CHW mainly just observes)"
                ],
                final_diagnosis="Persistent diarrhea (duration ≥14 days) – a danger sign indicating possible severe underlying issue (e.g. chronic infection or malnutrition)",
                treatment_plan="Refer to health facility for thorough evaluation. A child with ≥14 days of diarrhea needs medical assessment (possible causes like chronic infection or malabsorption) and likely specialized treatment. In the meantime, counsel the caregiver to continue feeding (the child is likely malnourished or at risk) and give fluids. The CHW can provide ORS and zinc to manage symptoms, but the key is referral for further care.",
                followup_instructions="Urgent referral. Explain to the caregiver that because the diarrhea has lasted so long, the child needs tests or treatments beyond community care. Ensure the caregiver understands to go to the facility in the next day or two at latest. After referral, the CHW should follow up (within a week) on the outcome and assist with nutritional support (for example, ensuring feeding plans are followed) once the child is back in the community."
            ),
            "diarrhea_vomiting_everything": GradingCriteria(
                diagnosis="Diarrhea with vomiting everything (unable to keep fluids down)",
                required_questions=[
                    "Ask: Is the child vomiting? If yes, does the child vomit everything he or she drinks or eats? (this determines if oral treatment is possible)",
                    "Ask: How long has the diarrhea been and how many times has the child vomited? (to gauge severity)"
                ],
                optional_questions=[
                    "Ask if vomit is immediately after drinking or only occasional (to confirm it's truly 'vomits everything')",
                    "Confirm no blood in stool (so that dysentery doesn't complicate management)"
                ],
                required_exams=[
                    "Observe the child drinking (if possible): if the child vomits everything offered (unable to hold down ORS/breastmilk), this is a danger sign – severe dehydration or another serious condition is likely",
                    "Check for other danger signs: is the child lethargic/unresponsive? (Often accompanies inability to retain fluids)",
                    "Check skin turgor and eyes (though if vomiting everything, severe dehydration is already implied)"
                ],
                final_diagnosis="Severe dehydration (or very severe illness): Child cannot keep fluids down ('vomits everything'), which is a danger sign requiring urgent referral",
                treatment_plan="Immediate referral: A child who vomits everything cannot rehydrate orally or take oral meds. Refer urgently to a facility for IV fluids or advanced care. If available, administer ORS via nasogastric tube or small sips and hope some stays in (if trained and feasible) while en route – but do not delay transport. Do not give zinc or antibiotic orally now, since the child will vomit it; these will be given at the facility. If the child has any signs of severe malnutrition or other illness, mention them in the referral note.",
                followup_instructions="Urgent hospital referral (same hour). Advise the caregiver that the child needs IV treatment and cannot be managed at home. Assist with arranging transport. There is no at-home follow-up since facility care is needed immediately. After stabilization at the hospital, the CHW should check in with the family within 1–2 days. For any future illness, advise caregivers to seek help early (since this episode escalated to severe dehydration)."
            ),
            "fever_acute": GradingCriteria(
                diagnosis="Fever (acute, <7 days)",
                required_questions=[
                    "Ask: How many days has the child had a fever? (distinguish from prolonged fever ≥7 days)",
                    "Ask if the child has had any convulsions during this illness (sign of possible severe malaria or other severe illness)",
                    "Confirm if the child has other symptoms like cough or diarrhea (to check for another cause of fever)"
                ],
                optional_questions=[
                    "Ask if the child was sleepier than usual or unconscious at any point (could indicate severe febrile illness)",
                    "Ask about rash or other findings (e.g. rash could indicate measles) – this doesn't change malaria treatment but is additional info"
                ],
                required_exams=[
                    "Feel for fever or measure temperature (confirm child is actually hot now)",
                    "Do a malaria rapid diagnostic test (RDT) since fever <7 days in a malaria area should be tested for malaria",
                    "General exam: check for any rash, ear infection, or throat redness (to identify non-malaria causes if RDT is negative, though CHW's action won't change much aside from referral if severe)",
                    "Check for danger signs: e.g. stiff neck (if trained), or lethargy. Also assess nutrition (MUAC) as usual"
                ],
                final_diagnosis="Febrile illness – suspect Malaria (pending test): acute fever with no localizing signs; assume malaria until proven otherwise in a malaria-endemic area",
                treatment_plan="If RDT is positive: Diagnosis = Malaria (confirmed). Start full course of recommended antimalarial (e.g. Artemisinin-based Combination Therapy). Give the first dose of antimalarial (such as Artemether-Lumefantrine) immediately and observe the child for a short while. If fever is high, you may also give antipyretic (e.g. paracetamol) to reduce fever and keep the child hydrated. If RDT is negative: No malaria detected. Do not give antimalarial; instead, advise supportive care: keep child hydrated, tepid sponging for fever, and continue monitoring. (The illness could be a viral infection or early malaria not caught by test.)",
                followup_instructions="Follow up in 3 days regardless of test result, to ensure the fever has resolved. If malaria, check if fever subsided and doses are being taken correctly; if not improving by day 3, refer to facility. If RDT-negative and the child still has fever after 3 days, this is now a persistent fever → refer for evaluation. In all cases, advise caregivers to return immediately if danger signs develop (convulsions, inability to drink, etc.)"
            ),
            "fever_malaria_confirmed": GradingCriteria(
                diagnosis="Fever with malaria (confirmed) (e.g. RDT positive)",
                required_questions=[
                    "Ask: When did the fever start? (to ensure it's not >7 days)",
                    "Ask if child has vomited anything (important because vomiting could affect oral medication – if vomits everything, that's a danger sign)"
                ],
                optional_questions=[
                    "Ask if the child was treated with a net or any antimalarial before coming (optional, for context)",
                    "Ask about any other family members ill (epidemiological interest, not altering immediate care)"
                ],
                required_exams=[
                    "Perform Malaria RDT (if not already done) – it is positive in this scenario",
                    "Assess for anemia (look at palm pallor) if trained, since malaria can cause anemia (optional)",
                    "Check temperature and general condition (ensure no other danger sign like lethargy or convulsions – if present, treat as severe malaria)"
                ],
                final_diagnosis="Malaria (uncomplicated) – fever with positive malaria test, no other severe signs",
                treatment_plan="Antimalarial treatment: Give full course of recommended ACT. For example, Artemether-Lumefantrine (AL): correct dose by age/weight, twice daily for 3 days. Administer the first dose now and observe the child for a short time. If the child vomits this dose, re-administer after a few minutes. If the fever is high, give paracetamol (if available) to reduce fever and keep the child comfortable. Encourage the caregiver to continue breastfeeding and fluids to prevent dehydration from fever.",
                followup_instructions="Follow up in 3 days to ensure the fever is gone and the full course of meds is being taken. At follow-up, check for fever or any new symptoms. If the child still has fever after completing treatment, or if condition worsens (e.g. develops weakness or persistent vomiting), refer to a health facility as this may indicate treatment failure or another illness. Advise caregiver to sleep under an insecticide-treated bednet to prevent future malaria (provide one if available)."
            ),
            "fever_negative_malaria": GradingCriteria(
                diagnosis="Fever with negative malaria test (no obvious source)",
                required_questions=[
                    "Ask: How long has the fever been present? (ensure it's <7 days so far)",
                    "Ask about any other symptoms (cough, ear pain, rash, etc. even if not initially reported) to look for another cause of fever"
                ],
                optional_questions=[
                    "Ask if the child was recently immunized for measles or has rash (if measles is possible, though CHW primarily will still refer if severe)",
                    "Ask if anyone else at home is ill (to consider if it could be an illness like flu)"
                ],
                required_exams=[
                    "Confirm RDT result is negative (no malaria) and double-check there are no danger signs (child is alert, drinking, no stiff neck)",
                    "General exam: check throat, ears, and skin for clues (e.g. red throat, ear discharge, rash). If an alternate cause like an ear infection is suspected and within CHW scope, manage accordingly (though in this manual, likely not, so mostly observational)"
                ],
                final_diagnosis="Fever (malaria test negative) – no immediate community diagnosis. Likely a viral infection or other mild illness, unless it persists",
                treatment_plan="Supportive care at home: Since malaria is ruled out and no other treatable focus is identified, do not give antimalarial. Advise the caregiver to manage fever by tepid sponging, keep the child lightly clothed, and ensure fluid intake. If the child is uncomfortable, they can be given a fever reducer (paracetamol) if available/appropriate. Advise continued feeding and hydration. There is no specific medication to give for a simple fever of unknown cause at CHW level.",
                followup_instructions="Follow up in 3 days to see if the fever has resolved. If by 3 days the child is fever-free and well, no further action is needed. If fever persists to day 7 or the child's condition worsens at any time, this becomes a danger sign – refer the child to a facility for evaluation. Instruct the caregiver to return immediately if any new symptoms develop (for example, rash, difficulty breathing, or refusal to eat/drink)."
            ),
            "fever_convulsions": GradingCriteria(
                diagnosis="Fever with convulsions (or other severe signs)",
                required_questions=[
                    "Ask the caregiver to describe the convulsion: how long it lasted, and if the child lost consciousness",
                    "Ask: Did the child have a fever when the convulsion occurred? (most febrile seizures are with fever – helps identify cause)",
                    "Ask if the child is able to drink now and behave normally after the convulsion"
                ],
                optional_questions=[
                    "If the child is conscious now: ask if there are any remaining neurological issues (weakness, confusion) – optional, as referral is needed regardless",
                    "Ask about any history of convulsions in the past without fever (to differentiate febrile seizure vs. epilepsy, though management at CHW level is same: referral)"
                ],
                required_exams=[
                    "Check for fever: measure temperature; if high, suspect febrile seizure (likely severe malaria or meningitis)",
                    "Assess level of consciousness: Is the child now unusually sleepy or still unconscious after convulsion? (Critical danger sign)",
                    "Look for neck stiffness or rash (if trained, to note if meningitis or other cause is possible)",
                    "Perform a malaria RDT immediately (convulsions + fever could be severe malaria) – if positive, it supports malaria as cause, but regardless, this is severe → refer"
                ],
                final_diagnosis="Severe febrile illness (e.g. Severe malaria or possible meningitis) – danger sign: convulsions with fever. Requires urgent referral (potentially life-threatening)",
                treatment_plan="Urgent pre-referral treatment: If malaria is suspected (fever present, malaria endemic area) give a rectal artesunate suppository before referral (especially if child is unconscious or convulsing). This helps start malaria treatment during transit. Do NOT give oral medications if the child has altered consciousness or is convulsing (risk of aspiration). If convulsion has stopped and child is conscious, you may give paracetamol for fever. Refer immediately to the hospital for advanced care (possible IV antimalarials, antiseizure medication, or antibiotics).",
                followup_instructions="Emergency referral – instruct caregiver to go to the hospital immediately. If convulsions are ongoing or repeat, this is a medical emergency. The CHW should accompany or ensure prompt transport. Little follow-up is done by CHW until the child is stabilized at the facility. After hospital care, the CHW should follow up with the family within a few days to support adherence to any ongoing treatments and to reinforce use of bednets (in case of malaria)."
            ),
            "fever_persistent": GradingCriteria(
                diagnosis="Fever ≥7 days (persistent fever)",
                required_questions=[
                    "Ask: Has the child had fever for a week or longer continuously? (confirm duration)",
                    "Ask if fever has been coming and going or present daily, and if any treatment was given (this is mainly for info; persistent fever is a danger sign regardless)"
                ],
                optional_questions=[
                    "Ask about any associated symptoms over the week (weight loss, cough, etc., to note if TB, HIV, or other chronic infection might be suspect)",
                    "Ask if the caregiver sought prior care for this prolonged fever (optional background)"
                ],
                required_exams=[
                    "Confirm persistent fever: measure temperature if possible; even if not febrile at the moment, the history of ≥7 days of fever episodes qualifies as danger sign",
                    "Examine for clues: enlarged lymph nodes, chronic cough, rash, etc. (CHW won't diagnose these, but notes if present for referral info)",
                    "Ensure malaria test was done (if not, do one, but after 7 days of fever even a negative test means referral, as per policy)"
                ],
                final_diagnosis="Persistent fever (≥7 days) – danger sign suggesting a serious illness (e.g. typhoid, TB, or other chronic infection) beyond CHW scope",
                treatment_plan="Refer to health facility for evaluation of persistent fever. Explain to caregiver that a fever lasting a week or more needs further tests (could be a more serious infection). Do not start new medications at community level (unless directed by prior guidance). If not already done, a malaria test can be repeated en route if fever is high, but essentially the child needs a doctor's assessment. Advise keeping the child hydrated and comfortable on the way; give paracetamol for fever if the child is in discomfort.",
                followup_instructions="Referral is the main plan. Advise caregiver to have the child seen at a health facility within a day. In the meantime, the caregiver should continue giving fluids and food. The CHW should follow up on the outcome: if possible, set a time to visit after a few days to see what the facility found and to help ensure any prescribed treatments (for example, for TB or other illness) are being followed. Remind the caregiver to return immediately if any new danger signs appear."
            ),
            "malnutrition_severe": GradingCriteria(
                diagnosis="Child looks very thin or has edema (possible malnutrition)",
                required_questions=[
                    "Ask the caregiver about the child's feeding: what foods and how often the child is eating (to assess diet quality)",
                    "Ask if the child has been losing weight or had any recent illness causing poor appetite"
                ],
                optional_questions=[
                    "Inquire about breastfeeding status (if under 2 years, is the child still breastfed?)",
                    "Ask if the family has difficulty obtaining food (optional, for context to provide support)"
                ],
                required_exams=[
                    "Measure MUAC (Mid-upper arm circumference): result is in the red zone (MUAC