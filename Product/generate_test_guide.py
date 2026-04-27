"""Generate a compressed ~2000-word WHO 2012 CHW guide equivalent for pipeline testing.

Preserves all clinical decision logic, danger signs, treatment protocols,
dosage tables, and follow-up rules from the original 141-page manual in a
compact format the RLM pipeline can extract from.
"""

from fpdf import FPDF


class WHOTestGuide(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 8)
        self.cell(
            0, 5,
            "Caring for the Sick Child in the Community (Compressed Test Edition)",
            align="C",
        )
        self.ln(6)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title: str):
        self.set_font("Helvetica", "B", 13)
        self.set_fill_color(220, 230, 241)
        self.cell(0, 8, title, ln=True, fill=True)
        self.ln(2)

    def sub_title(self, title: str):
        self.set_font("Helvetica", "B", 11)
        self.cell(0, 7, title, ln=True)
        self.ln(1)

    def body_text(self, text: str):
        self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 5, text)
        self.ln(2)

    def bullet(self, text: str):
        self.set_font("Helvetica", "", 10)
        self.cell(5, 5, "-")
        self.multi_cell(0, 5, text)
        self.ln(1)

    def table_row(self, cells: list[str], widths: list[int], bold: bool = False):
        self.set_font("Helvetica", "B" if bold else "", 9)
        h = 6
        for i, cell in enumerate(cells):
            self.cell(widths[i], h, cell, border=1)
        self.ln(h)


def build_pdf():
    pdf = WHOTestGuide()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # ── Title page ──
    pdf.add_page()
    pdf.ln(30)
    pdf.set_font("Helvetica", "B", 22)
    pdf.cell(0, 12, "Caring for the Sick Child", align="C", ln=True)
    pdf.cell(0, 12, "in the Community", align="C", ln=True)
    pdf.ln(8)
    pdf.set_font("Helvetica", "", 14)
    pdf.cell(0, 8, "Participant Manual", align="C", ln=True)
    pdf.cell(0, 8, "Compressed Test Edition (~2 000 words)", align="C", ln=True)
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, "Based on: WHO/UNICEF 2012", align="C", ln=True)
    pdf.cell(0, 7, "ISBN 978 92 4 154804 5", align="C", ln=True)
    pdf.ln(15)
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(0, 5, (
        "This compressed edition preserves all clinical decision logic, danger signs, "
        "treatment protocols, dosage tables, and follow-up rules from the original "
        "141-page manual. Exercises, role plays, photos, and facilitation notes are "
        "omitted. For training use only with the full manual."
    ), align="C")

    # ── Page 2: Scope & Recording Form ──
    pdf.add_page()

    pdf.section_title("1. INTRODUCTION")
    pdf.body_text(
        "This manual trains community health workers (CHWs) to assess, classify, "
        "and treat sick children age 2 months up to 5 years in the community, as "
        "part of the Integrated Management of Childhood Illness (IMCI) strategy. "
        "CHWs identify signs of illness, treat diarrhoea, confirmed malaria, and "
        "pneumonia (cough with fast breathing) at home, and urgently refer children "
        "with danger signs to a health facility."
    )

    pdf.section_title("2. SICK CHILD RECORDING FORM")
    pdf.body_text(
        "For every visit, the CHW completes the Sick Child Recording Form. "
        "Record: date, CHW name, child's name, age (years and months), sex, "
        "caregiver's name and relationship (mother/father/other), and address. "
        "The form guides the visit: identify problems, decide to refer or treat, "
        "give medicine, advise on home care, and schedule follow-up."
    )

    # ── Page 3: Identify Problems ──
    pdf.section_title("3. IDENTIFY PROBLEMS")
    pdf.body_text(
        "ASK the caregiver about the child's problems, then LOOK for signs of "
        "illness. The findings determine whether to refer or treat at home."
    )

    pdf.sub_title("3.1 ASK: What are the child's problems?")

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "Cough", ln=True)
    pdf.body_text(
        "Ask: Does the child have a cough? If yes, for how long? "
        "A cough for 14 days or more is a danger sign."
    )

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "Diarrhoea", ln=True)
    pdf.body_text(
        "Diarrhoea is 3 or more loose or watery stools in 24 hours. "
        "Ask: For how long? Is there blood in the stool? "
        "Diarrhoea for 14 days or more is a danger sign. "
        "Blood in stool (dysentery) is a danger sign."
    )

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "Fever", ln=True)
    pdf.body_text(
        "Ask: Does the child have fever now or in the last 3 days? "
        "If yes, when did it start? Fever for 7 days or more is a danger sign. "
        "Identify fever by the caregiver's report or by feeling the child's "
        "stomach or underarm."
    )

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "Convulsions", ln=True)
    pdf.body_text(
        "A convulsion during the current illness is a danger sign. "
        "The child needs urgent referral."
    )

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "Not able to drink or feed", ln=True)
    pdf.body_text(
        "A child who cannot drink or feed anything at all has a danger sign. "
        "Ask the caregiver to offer the child a drink or breastfeed."
    )

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "Vomiting everything", ln=True)
    pdf.body_text(
        "A child who vomits everything and cannot hold anything down has a "
        "danger sign. A child who vomits sometimes but can hold down some fluids "
        "does not have this sign."
    )

    pdf.sub_title("3.2 LOOK: Signs of illness on examination")

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "Chest indrawing", ln=True)
    pdf.body_text(
        "Look at the lower chest wall when the child breathes IN. "
        "Chest indrawing is when the lower chest wall goes IN when the child "
        "breathes in. This is a danger sign indicating severe pneumonia. "
        "The child must be calm; do not assess while crying or breastfeeding."
    )

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "Fast breathing", ln=True)
    pdf.body_text(
        "Count the child's breaths for one full minute while the child is "
        "calm and quiet. Fast breathing thresholds by age:"
    )

    w = [90, 90]
    pdf.table_row(["Age group", "Fast breathing if"], w, bold=True)
    pdf.table_row(["2 months up to 12 months", "50 breaths per minute or more"], w)
    pdf.table_row(["12 months up to 5 years", "40 breaths per minute or more"], w)
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "Unusually sleepy or unconscious", ln=True)
    pdf.body_text(
        "A child who is unusually sleepy, difficult to wake, or unconscious "
        "has a danger sign. Try to wake the child by clapping or speaking loudly."
    )

    # ── Page 4: Malnutrition ──
    pdf.section_title("3.3 LOOK for signs of severe malnutrition")
    pdf.body_text(
        "Use the MUAC (Mid-Upper Arm Circumference) strap on children age "
        "6 months up to 5 years. Wrap the strap around the child's left upper arm, "
        "midway between shoulder and elbow. Read the colour:"
    )
    pdf.bullet("GREEN: not malnourished")
    pdf.bullet("YELLOW: moderate malnutrition (refer to supplementary feeding if available)")
    pdf.bullet("RED: severe malnutrition - DANGER SIGN - refer urgently")

    pdf.body_text(
        "Also check for swelling of both feet (oedema). Press thumbs gently on "
        "the top of each foot for 3 seconds. If dents remain on BOTH feet, the "
        "child has severe malnutrition. Refer urgently."
    )

    # ── Danger signs summary ──
    pdf.section_title("4. DECIDE: REFER OR TREAT")

    pdf.sub_title("4.1 Danger signs requiring URGENT referral")
    pdf.body_text("If ANY of these danger signs are present, refer the child urgently:")

    dangers = [
        "Cough for 14 days or more",
        "Diarrhoea for 14 days or more",
        "Blood in stool",
        "Fever for 7 days or more",
        "Convulsions",
        "Not able to drink or feed",
        "Vomits everything",
        "Chest indrawing (severe pneumonia)",
        "Unusually sleepy or unconscious",
        "Red on MUAC strap (severe acute malnutrition)",
        "Swelling of both feet (oedema)",
    ]
    for d in dangers:
        pdf.bullet(d)

    pdf.body_text(
        "A child with any danger sign is too ill to treat in the community. "
        "Give the first dose of medicine before referral if the child can drink. "
        "Write a referral note. Arrange transportation."
    )

    # ── Treatment at home ──
    pdf.sub_title("4.2 No danger sign: TREAT AT HOME")
    pdf.body_text(
        "Children without danger signs who have diarrhoea, fever (in a malaria "
        "area), or cough with fast breathing may be treated at home."
    )

    # ── Diarrhoea treatment ──
    pdf.section_title("5. TREATMENT PROTOCOLS")

    pdf.sub_title("5.1 Diarrhoea (less than 14 days, no blood, no danger sign)")
    pdf.body_text("Give ORS (Oral Rehydration Salts) and zinc supplement.")

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "ORS preparation and administration:", ln=True)
    pdf.body_text(
        "1. Wash hands with soap and water. "
        "2. Pour entire ORS packet into a clean container. "
        "3. Add 1 litre of clean water and mix until dissolved. "
        "4. Give frequent small sips from a cup or spoon. "
        "5. If the child vomits, wait 10 minutes, then give more slowly. "
        "6. Continue breastfeeding. Give at least half a cup after each loose stool. "
        "7. Store mixed ORS in a covered container; discard after 24 hours."
    )

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "Zinc supplement dosage:", ln=True)
    pdf.ln(2)

    w3 = [55, 40, 35, 50]
    pdf.table_row(["Age", "Dose", "Frequency", "Duration / Supply"], w3, bold=True)
    pdf.table_row(["2 months up to 6 months", "1/2 tablet", "Once daily", "10 days / 5 tablets"], w3)
    pdf.table_row(["6 months up to 5 years", "1 tablet", "Once daily", "10 days / 10 tablets"], w3)
    pdf.ln(2)
    pdf.body_text(
        "Dissolve the zinc tablet (or half tablet) in breast milk or clean water "
        "on a spoon. Give the full 10 days even if diarrhoea stops. Zinc reduces "
        "severity, shortens duration, and prevents diarrhoea for up to 3 months."
    )

    # ── Fever / Malaria treatment ──
    pdf.sub_title("5.2 Fever (less than 7 days) in a malaria area")
    pdf.body_text(
        "Do a Rapid Diagnostic Test (RDT) for malaria before treating. "
        "Do not give antimalarial medicine unless the RDT is positive."
    )

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "RDT procedure (summary):", ln=True)
    pdf.body_text(
        "1. Check expiry date; do not use expired tests. "
        "2. Wear new gloves for each child. "
        "3. Clean the child's ring finger with a spirit swab; let dry. "
        "4. Prick the finger with a lancet; collect blood with the loop. "
        "5. Place blood in square hole A on the test strip. "
        "6. Add 6 drops of buffer into round hole B. "
        "7. Wait 15 minutes. "
        "8. Read results: POSITIVE = red lines in both C and T windows. "
        "NEGATIVE = red line in C window only. "
        "INVALID = no line in C window; repeat with new test."
    )

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "Antimalarial AL (Artemether-Lumefantrine) dosage:", ln=True)
    pdf.ln(2)

    w4 = [55, 40, 35, 50]
    pdf.table_row(["Age", "Dose", "Frequency", "Duration / Supply"], w4, bold=True)
    pdf.table_row(["2 months up to 3 years", "1 tablet AL", "Twice daily", "3 days / 6 tablets"], w4)
    pdf.table_row(["3 years up to 5 years", "2 tablets AL", "Twice daily", "3 days / 12 tablets"], w4)
    pdf.ln(2)
    pdf.body_text(
        "Give first dose immediately. Second dose 8 hours later. Then morning "
        "and evening for 2 more days. Crush tablet and mix with breast milk, "
        "water, or food. If the child spits out the entire dose, give another. "
        "If unable to take medicine, refer to health facility."
    )

    pdf.body_text(
        "If RDT is negative: do not give antimalarial. Follow up in 3 days. "
        "If fever persists at follow-up, refer to health facility."
    )

    # ── Pneumonia treatment ──
    pdf.sub_title("5.3 Cough with fast breathing (pneumonia, no chest indrawing)")
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "Oral amoxicillin dosage (250 mg tablets):", ln=True)
    pdf.ln(2)

    pdf.table_row(["Age", "Dose", "Frequency", "Duration / Supply"], w4, bold=True)
    pdf.table_row(["2 months up to 12 months", "1 tablet", "Twice daily", "5 days / 10 tablets"], w4)
    pdf.table_row(["12 months up to 5 years", "2 tablets", "Twice daily", "5 days / 20 tablets"], w4)
    pdf.ln(2)
    pdf.body_text(
        "Check expiration date. Crush the tablet and mix with breast milk or "
        "water. Give the first dose in front of you. Give the full 5 days even "
        "if the child feels better."
    )

    # ── Home care ──
    pdf.section_title("6. HOME CARE ADVICE (for all children treated at home)")
    pdf.bullet("Give more fluids and continue feeding.")
    pdf.bullet("Return immediately if the child: cannot drink or feed, becomes sicker, or has blood in stool.")
    pdf.bullet("Sleep under an insecticide-treated bednet (ITN).")
    pdf.bullet("Keep all medicine out of reach of children.")
    pdf.bullet("Store medicine in a clean, dry place.")
    pdf.ln(2)

    # ── Follow-up ──
    pdf.section_title("7. FOLLOW-UP VISITS")
    pdf.body_text(
        "Follow up ALL children treated at home in 3 days. If the child is not "
        "better or is worse at follow-up, refer to a health facility."
    )

    pdf.sub_title("7.1 Follow-up for diarrhoea")
    pdf.body_text(
        "After 3 days: Is the child still having diarrhoea? If yes and for "
        "less than 14 days, continue ORS and zinc. If 14 days or more, refer. "
        "Ensure the caregiver is giving zinc for the full 10 days."
    )

    pdf.sub_title("7.2 Follow-up for fever / malaria")
    pdf.body_text(
        "After 3 days: Does the child still have fever? If RDT was positive and "
        "antimalarial was given, the child should improve. If still febrile, "
        "refer. If RDT was negative and fever persists, refer."
    )

    pdf.sub_title("7.3 Follow-up for cough with fast breathing")
    pdf.body_text(
        "After 3 days: Is the child still coughing? Count breaths again. "
        "If still fast breathing, refer. If breathing is normal and child is "
        "improving, complete the full 5-day course of amoxicillin."
    )

    # ── Referral ──
    pdf.section_title("8. REFERRAL PROCEDURES")
    pdf.body_text(
        "When referring a child: "
        "1. Explain to the caregiver why the child must go to the health facility. "
        "2. Write a referral note including: child's name, age, date, reason for "
        "referral, danger signs found, treatment already given, and your name. "
        "3. Give the first dose of any appropriate medicine before referral. "
        "4. Help arrange transportation. "
        "5. If fever, keep the child cool during travel. "
        "6. If diarrhoea, give ORS solution during travel. "
        "7. Follow up the child after return, at least once a week until well."
    )

    # ── Supply list ──
    pdf.section_title("9. CHW MEDICINE KIT AND SUPPLIES")
    pdf.body_text("The CHW must maintain the following supplies:")

    pdf.sub_title("Medicines")
    pdf.bullet("ORS packets")
    pdf.bullet("Zinc tablets (20 mg dispersible)")
    pdf.bullet("Artemether-Lumefantrine (AL) tablets")
    pdf.bullet("Amoxicillin tablets (250 mg)")

    pdf.sub_title("Diagnostic equipment")
    pdf.bullet("Rapid Diagnostic Tests (RDT) for malaria")
    pdf.bullet("MUAC straps")
    pdf.bullet("Timer or watch with second hand (for counting breaths)")
    pdf.bullet("Thermometer (optional)")

    pdf.sub_title("Other supplies")
    pdf.bullet("Gloves (new pair for each RDT)")
    pdf.bullet("Lancets")
    pdf.bullet("Spirit swabs (alcohol swabs)")
    pdf.bullet("Sick Child Recording Forms")
    pdf.bullet("Referral note forms")
    pdf.bullet("Buffer solution (for RDTs)")
    pdf.bullet("Non-sharps waste container")

    # ── Back page ──
    pdf.add_page()
    pdf.ln(20)
    pdf.set_font("Helvetica", "I", 10)
    pdf.multi_cell(0, 6, (
        "This compressed test edition was created for automated pipeline testing. "
        "It preserves the clinical decision logic from: WHO/UNICEF (2012) "
        "Caring for the Sick Child in the Community. "
        "ISBN 978 92 4 154804 5. "
        "All clinical thresholds, dosage tables, danger sign classifications, "
        "and treatment protocols are faithful to the original manual."
    ), align="C")

    out_path = "WHO_CHW_guide_2012_test.pdf"
    pdf.output(out_path)
    return out_path


if __name__ == "__main__":
    path = build_pdf()
    print(f"Generated: {path}")

    # Word count estimate
    import fitz
    doc = fitz.open(path)
    total_text = " ".join(doc[i].get_text() for i in range(doc.page_count))
    words = len(total_text.split())
    print(f"Pages: {doc.page_count}")
    print(f"Approx word count: {words}")
