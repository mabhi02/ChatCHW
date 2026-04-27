// translator.js 
// Lookup-table layer between patient data and the CHT test harness.

// Takes a patient object with named fields and produces a flat
// answer array that the harness's fillForm function expects.
// The lookup table mirrors the orchestrator form's DOM order*.
// Each entry: { field, type }
//   - field:   the key to look up in the patient record
//   - type:    'string' (passed as is) or 'int' (converted to a JS number)
// *Conditional follow-ups are handled separately; they get inserted
// into the array only when their parent field is "yes".
// Missing fields throw errors, zero defaults/assumptions.

const FORM_FIELDS = [
  // patient_info
  { field: 'child_name',                          type: 'string' },
  { field: 'age_months',                          type: 'int' },
  { field: 'sex',                                 type: 'string' },

  // danger_signs
  { field: 'convulsions_present',                 type: 'string' },
  { field: 'unusually_sleepy_or_unconscious',     type: 'string' },
  { field: 'not_able_to_drink_or_feed_anything',  type: 'string' },
  { field: 'vomits_everything',                   type: 'string' },
  { field: 'chest_indrawing',                     type: 'string' },
  { field: 'swelling_of_both_feet',               type: 'string' },
  { field: 'muac_color',                          type: 'string' },
];

const CONDITIONAL_BLOCKS = [
  {
    parent: { field: 'cough_present',      type: 'string' },
    followUps: [
      { field: 'cough_duration_days',      type: 'int' },
      { field: 'breaths_per_minute',       type: 'int' },
    ]
  },
  {
    parent: { field: 'diarrhoea_present',  type: 'string' },
    followUps: [
      { field: 'diarrhoea_duration_days',  type: 'int' },
      { field: 'blood_in_stool',           type: 'string' },
    ]
  },
  {
    parent: { field: 'fever_present',      type: 'string' },
    followUps: [
      { field: 'fever_duration_days',      type: 'int' },
      { field: 'rdt_result',              type: 'string' },
    ]
  },
];

// Pull a value from the patient record, throw if missing/invalid.
function getValue(patient, entry) {
  const raw = patient[entry.field];

  if (raw === undefined || raw === null || raw === '') {
    throw new Error(`Missing required field: ${entry.field}`);
  }

  if (entry.type === 'int') {
    const n = Number(raw);
    if (isNaN(n)) {
      throw new Error(`Invalid number for field ${entry.field}: "${raw}"`);
    }
    return n;
  }
  return raw;
}

// Takes a patient object with named fields, returns the flat answer array.
// Throws on any missing or invalid field, no defaults.
function translate(patient) {
  const answers = [];

  // Fixed fields (positions 1-10)
  for (const entry of FORM_FIELDS) {
    answers.push(getValue(patient, entry));
  }

  // Conditional blocks (cough, diarrhoea, fever)
  for (const block of CONDITIONAL_BLOCKS) {
    const parentValue = getValue(patient, block.parent);
    answers.push(parentValue);

    if (parentValue === 'yes') {
      // Parent is yes: follow-up fields are required and included
      for (const followUp of block.followUps) {
        answers.push(getValue(patient, followUp));
      }
    }
    // Parent is no: follow-ups are not expected and not included
  }
  return answers;
}

module.exports = { translate };
