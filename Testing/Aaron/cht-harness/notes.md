# Dependencies

- Node.js 22
- npm packages installed in the project folder: 
  `cht-conf-test-harness`, `chai`, `mocha`
- The orchestrator form as XML at `forms/app/orchestrator.xml` 
- `app_settings.json` at the project root (the harness 
  requires this file to exist even though our prototype doesn't 
  read from it)

From cht-harness, run:

npx mocha test.js --timeout 30000

# Answer structure note

Learned that the harness renders the entire orchestrator form as a single page.
Section tags in the XML (patient_info, danger_signs, respiratory, etc.)
are visual groupings, not separate pages, and all 13 baseline questions must
be passed as one flat array in form order. And only in cases where symptoms are present,
follow up questions are asked:

Max 19-Question Case:
|  # | Field | Datatype | Valid values | When to include |

|  1 | child_name | string | any | always |
|  2 | age_months | int | 0–59 | always |
|  3 | sex | string | male / female | always |
|  4 | convulsions_present | string | yes / no | always |
|  5 | unusually_sleepy_or_unconscious | string | yes / no | always |
|  6 | not_able_to_drink_or_feed_anything | string | yes / no | always |
|  7 | vomits_everything | string | yes / no | always |
|  8 | chest_indrawing | string | yes / no | always |
|  9 | swelling_of_both_feet | string | yes / no | always |
| 10 | muac_color | string | red / yellow / green | always |
| 11 | cough_present | string | yes / no | always |
| 12 | cough_duration_days | int | ≥ 0 | only if #11 = yes |
| 13 | breaths_per_minute | int | > 0 | only if #11 = yes |
| 14 | diarrhoea_present | string | yes / no | always |
| 15 | diarrhoea_duration_days | int | ≥ 0 | only if #14 = yes |
| 16 | blood_in_stool | string | yes / no | only if #14 = yes |
| 17 | fever_present | string | yes / no | always |
| 18 | fever_duration_days | int | ≥ 0 | only if #17 = yes |
| 19 | rdt_result | string | positive / negative | only if #17 = yes |

Example of Healthy/No Symptoms:
['Child A', 24, 'male',
 'no', 'no', 'no', 'no', 'no', 'no', 'green',
 'no',
 'no',
 'no']

Example of Cough Only:
['Child B', 24, 'male',
 'no', 'no', 'no', 'no', 'no', 'no', 'green',
 'yes', 3, 45,
 'no',
 'no']

Diarrhea Only:
 ['Child C', 24, 'male',
 'no', 'no', 'no', 'no', 'no', 'no', 'green',
 'no',
 'yes', 2, 'no',
 'no']

Fever Only:
['Child D', 24, 'male',
 'no', 'no', 'no', 'no', 'no', 'no', 'green',
 'no',
 'no',
 'yes', 4, 'negative']

Example of Cough + Diarrhea:
['Child E', 24, 'male',
 'no', 'no', 'no', 'no', 'no', 'no', 'green',
 'yes', 3, 45,
 'yes', 2, 'no',
 'no']

Cough + Fever:
['Child F', 24, 'male',
 'no', 'no', 'no', 'no', 'no', 'no', 'green',
 'yes', 3, 45,
 'no',
 'yes', 4, 'negative']

Diarrhea + Fever:
['Child G', 24, 'male',
 'no', 'no', 'no', 'no', 'no', 'no', 'green',
 'no',
 'yes', 2, 'no',
 'yes', 4, 'negative']

Example of Cough + Diarrhea + Fever:
['Child H', 24, 'male',
 'no', 'no', 'no', 'no', 'no', 'no', 'green',
 'yes', 5, 52,
 'yes', 3, 'no',
 'yes', 2, 'positive']

# Blockers/Assumptions

Relies on a specific sequence of questions
Abstracted the translation layer
CHT only runs JS code

Since the CHT harness also relies on the inputted XLSForm, 
would having different forms for each case make sense?
Ideally the order of questions being asked can be arbitrary,
but when would we need to ask questions out of order?
