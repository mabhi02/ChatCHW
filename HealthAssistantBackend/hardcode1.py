# CHWBot: WHO 2024 Diarrhea & Pneumonia Decision Tree
# Based on WHO Guidelines on Management of Childhood Illness (IMCI), WHO 2024
# Citation: WHO. Guideline on management of pneumonia and diarrhoea in children up to 10 years of age. Geneva: WHO; 2024.

class CHWBot:
    def __init__(self):
        self.answers = {}
        self.diagnosis = []
        self.treatment = []
        self.see_why = []
        self.questions = [
            ("age", "What is the age of the child?", ["<2 months", "2–59 months", "5–9 years"]),
            ("danger_signs", "Any danger signs (convulsions, unable to drink, vomiting everything, lethargy)?", ["Yes", "No"]),
            ("cough", "Is there cough or difficulty breathing?", ["Yes", "No"]),
            ("breathing", "Is breathing fast or is chest moving inwards?", ["Fast breathing", "Chest indrawing", "None"]),
            ("diarrhea", "Does the child have diarrhea?", ["Yes", "No"]),
            ("diarrhea_details", "Duration and blood in stool?", ["<14 days", ">14 days", "Bloody stool", "None"]),
            ("dehydration", "Any signs of dehydration (sunken eyes, poor skin pinch)?", ["Severe", "Some", "None"]),
            ("supplies", "Which supplies are available?", ["ORS", "Zinc", "Amoxicillin", "None"]),
            ("fever", "Does the child have fever?", ["Yes", "No"]),
            ("fluid", "Is the child able to take fluids orally?", ["Yes", "No"]),
        ]

    def ask_questions(self):
        for q_id, question, choices in self.questions:
            print(f"\n{question}")
            for i, choice in enumerate(choices):
                print(f"{i+1}. {choice}")
            ans = input("Select option: ")
            try:
                self.answers[q_id] = choices[int(ans)-1]
            except:
                self.answers[q_id] = "Don't know"

    def diagnose(self):
        a = self.answers
        # 1. Danger signs
        if a["danger_signs"] == "Yes" or a["fluid"] == "No":
            self.diagnosis.append("Possible Severe Disease")
            self.treatment.append("Refer immediately to hospital")
            self.see_why.append("WHO 2024, p.10: Danger signs or inability to drink = urgent referral.")
            return

        # 2. Pneumonia logic
        if a["cough"] == "Yes":
            if a["breathing"] == "Fast breathing" and a["age"] == "2–59 months":
                self.diagnosis.append("Pneumonia")
                if "Amoxicillin" in a["supplies"]:
                    self.treatment.append("Give oral amoxicillin for 3–5 days")
                else:
                    self.treatment.append("Refer due to amoxicillin stockout")
                self.see_why.append("WHO 2024, p.10: Treat pneumonia with oral amoxicillin")
            elif a["breathing"] == "Chest indrawing" and a["age"] == "2–59 months":
                self.diagnosis.append("Chest Indrawing Pneumonia")
                if "Amoxicillin" in a["supplies"]:
                    self.treatment.append("Give oral amoxicillin for 5 days")
                else:
                    self.treatment.append("Refer due to amoxicillin stockout")
                self.see_why.append("WHO 2024, p.10: Chest indrawing pneumonia = oral amoxicillin")

        # 3. Diarrhea logic
        if a["diarrhea"] == "Yes":
            if a["diarrhea_details"] == "Bloody stool":
                self.diagnosis.append("Dysentery")
                self.treatment.append("Give antibiotics for dysentery")
                self.see_why.append("WHO 2024, p.19: Blood in stool = treat with antibiotics")
            elif a["diarrhea_details"] == "<14 days":
                self.diagnosis.append("Acute Watery Diarrhea")
                if "ORS" in a["supplies"]:
                    self.treatment.append("Give low-osmolarity ORS")
                else:
                    self.treatment.append("Refer due to ORS stockout")
                if "Zinc" in a["supplies"]:
                    self.treatment.append("Give oral zinc for 14 days")
                self.see_why.append("WHO 2024, p.19: Acute diarrhea = ORS + zinc")
            elif a["diarrhea_details"] == ">14 days":
                self.diagnosis.append("Persistent Diarrhea")
                if "ORS" in a["supplies"]:
                    self.treatment.append("Give ORS + zinc")
                else:
                    self.treatment.append("Refer due to ORS stockout")
                self.see_why.append("WHO 2024, p.19: Persistent diarrhea = ORS + zinc")

        # 4. Dehydration logic
        if a["dehydration"] == "Severe":
            self.diagnosis.append("Severe Dehydration")
            self.treatment.append("Refer urgently and start rehydration if possible")
            self.see_why.append("WHO 2024, p.26: Severe dehydration = urgent referral")
        elif a["dehydration"] == "Some":
            self.diagnosis.append("Some Dehydration")
            if "ORS" in a["supplies"]:
                self.treatment.append("Give ORS and monitor")
            else:
                self.treatment.append("Refer due to ORS stockout")
            self.see_why.append("WHO 2024, p.26: Some dehydration = ORS")

    def show_results(self):
        print("\n--- Diagnosis Result ---")
        for d in self.diagnosis:
            print("Diagnosis:", d)
        print("\n--- Treatment Plan ---")
        for t in self.treatment:
            print("Treatment:", t)
        print("\n--- Explanation ---")
        for e in self.see_why:
            print("See Why:", e)


if __name__ == "__main__":
    bot = CHWBot()
    bot.ask_questions()
    bot.diagnose()
    bot.show_results()
