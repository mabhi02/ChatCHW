// test.js
// Orchestrator Harness Prototype

// Loads the orchestrator XLSForm via the CHT test harness, runs patients
// through the full form, and captures the output in an execution log
// consistent with Emett's schema.
// Log can be found in harness_execution_log.json after running the tests.

const { expect } = require('chai');
const Harness = require('cht-conf-test-harness');
const fs = require('fs');
const { translate } = require('./translator');

const harness = new Harness({ verbose: true });

// Convert orchestrator's "yes"/"no" and "true"/"false" strings
// into actual booleans for Emett's schema
function toBool(val) {
  return val === 'yes' || val === 'true';
}

// Convert string numbers to actual numbers, empty/missing to null
function toNumberOrNull(val) {
  if (val === '' || val === undefined || val === null) return null;
  const n = Number(val);
  return isNaN(n) ? null : n;
}

// Converts the harness result into Emett's log schema
function toLogEntry(patientId, harnessResult) {
  // If the harness reported errors, log as failed execution
  if (harnessResult.errors && harnessResult.errors.length > 0) {
    return {
      timestamp: new Date().toISOString(),
      patient_id: patientId,
      inputs: {},
      execution_trace: {
        engine: 'xlsform_harness',
        form_name: 'orchestrator',
        status: 'error',
        errors: harnessResult.errors
      },
      final_outcome: null
    };
  }

  // On success, harness returns a report object with fields
  // sub-object containing everything the orchestrator form calculated
  const f = harnessResult.report.fields;

  return {
    timestamp: new Date().toISOString(),
    patient_id: patientId,

    // Inputs block contains typed values matching Emett's schema
    // Booleans for yes/no fields, numbers for durations, strings for enums
    inputs: {
      age_months: toNumberOrNull(f.patient_info.age_months),
      cough_present: toBool(f.respiratory.cough_present),
      cough_duration_days: toNumberOrNull(f.respiratory.cough_duration_days),
      fast_breathing_present: toBool(f.respiratory.fast_breathing_calc),
      chest_indrawing_present: toBool(f.danger_signs.chest_indrawing),
      has_diarrhoea: toBool(f.diarrhoea.diarrhoea_present),
      diarrhoea_duration_days: toNumberOrNull(f.diarrhoea.diarrhoea_duration_days),
      blood_in_stool: toBool(f.diarrhoea.blood_in_stool),
      rdt_result: f.fever.rdt_result || 'not_done',
      muac_result: f.danger_signs.muac_color || 'green'
    },

    execution_trace: {
      engine: 'xlsform_harness',
      form_name: 'orchestrator',
      status: 'success'
    },

    final_outcome: {
      triage: f.final_triage.toLowerCase(),
      danger_sign: toBool(f.danger_signs.danger_sign_present_flag),
      clinic_referral: toBool(f.clinic_referral_flag),
      reason: f.triage_reason_code.toLowerCase(),
      ref: '',           // orchestrator form doesn't produce a ref code
      actions: f.final_advice
    }
  };
}

describe('Orchestrator harness prototype', () => {

  // Shared log across all tests in this suite
  const executionLog = [];

  // Harness setup: open the virtual browser before tests, close it after, reset between each
  before(async () => { return await harness.start(); });
  after(async () => {
    // Write the full collected log after all tests finish
    fs.writeFileSync(
      'harness_execution_log.json',
      JSON.stringify(executionLog, null, 2)
    );
    return await harness.stop();
  });
  beforeEach(async () => { return await harness.clear(); });

  it('healthy patient → home', async () => {
    // Patient record with named fields — translator converts to flat array
    const patient = {
      child_name: 'Test Child',
      age_months: 24,
      sex: 'male',
      convulsions_present: 'no',
      unusually_sleepy_or_unconscious: 'no',
      not_able_to_drink_or_feed_anything: 'no',
      vomits_everything: 'no',
      chest_indrawing: 'no',
      swelling_of_both_feet: 'no',
      muac_color: 'green',
      cough_present: 'no',
      diarrhoea_present: 'no',
      fever_present: 'no'
    };

    const answers = translate(patient);
    const result = await harness.fillForm('orchestrator', answers);
    const logEntry = toLogEntry('row_1', result);
    executionLog.push(logEntry);

    console.log('\n------------ Healthy Patient ------------');
    console.log(JSON.stringify(logEntry, null, 2));
    console.log('-----------------------------------------\n');

    // Verify that the orchestrator ran and produced the expected triage
    expect(result.errors).to.be.empty;
    expect(logEntry.final_outcome.triage).to.equal('home');
  });

  it('danger sign (chest indrawing) → hospital', async () => {
    // Patient with a danger sign (chest indrawing), should route to HOSPITAL
    const patient = {
      child_name: 'Sick Child',
      age_months: 18,
      sex: 'female',
      convulsions_present: 'no',
      unusually_sleepy_or_unconscious: 'no',
      not_able_to_drink_or_feed_anything: 'no',
      vomits_everything: 'no',
      chest_indrawing: 'yes',
      swelling_of_both_feet: 'no',
      muac_color: 'green',
      cough_present: 'no',
      diarrhoea_present: 'no',
      fever_present: 'no'
    };

    const answers = translate(patient);
    const result = await harness.fillForm('orchestrator', answers);
    const logEntry = toLogEntry('row_2', result);
    executionLog.push(logEntry);

    console.log('\n------------ Danger Sign Patient ------------');
    console.log(JSON.stringify(logEntry, null, 2));
    console.log('---------------------------------------------\n');

    expect(result.errors).to.be.empty;
    expect(logEntry.final_outcome.triage).to.equal('hospital');
    expect(logEntry.final_outcome.danger_sign).to.equal(true);
  });

  it('cough + fast breathing → clinic', async () => {
    // Patient with cough for 3 days, 45 breaths/min
    // Age > 12 months + breaths > 40 triggers FAST_BREATHING, should route to CLINIC
    const patient = {
      child_name: 'Cough Child',
      age_months: 24,
      sex: 'male',
      convulsions_present: 'no',
      unusually_sleepy_or_unconscious: 'no',
      not_able_to_drink_or_feed_anything: 'no',
      vomits_everything: 'no',
      chest_indrawing: 'no',
      swelling_of_both_feet: 'no',
      muac_color: 'green',
      cough_present: 'yes',
      cough_duration_days: 3,
      breaths_per_minute: 45,
      diarrhoea_present: 'no',
      fever_present: 'no'
    };

    const answers = translate(patient);
    const result = await harness.fillForm('orchestrator', answers);
    const logEntry = toLogEntry('row_3', result);
    executionLog.push(logEntry);

    console.log('\n------------ Cough Patient ------------');
    console.log(JSON.stringify(logEntry, null, 2));
    console.log('---------------------------------------\n');

    expect(result.errors).to.be.empty;
    expect(logEntry.final_outcome.triage).to.equal('clinic');
    expect(logEntry.final_outcome.clinic_referral).to.equal(true);
    expect(logEntry.final_outcome.reason).to.equal('fast_breathing');
  });

  it('out-of-order patient data → translator handles correctly', async () => {
    // Fields are deliberately scrambled — translator should still
    // produce the correct positional array
    const patient = {
      fever_present: 'no',
      sex: 'female',
      cough_present: 'yes',
      chest_indrawing: 'no',
      age_months: 10,
      breaths_per_minute: 55,
      muac_color: 'green',
      child_name: 'Scrambled Child',
      vomits_everything: 'no',
      cough_duration_days: 2,
      swelling_of_both_feet: 'no',
      diarrhoea_present: 'no',
      convulsions_present: 'no',
      unusually_sleepy_or_unconscious: 'no',
      not_able_to_drink_or_feed_anything: 'no',
    };

    const answers = translate(patient);
    const result = await harness.fillForm('orchestrator', answers);
    const logEntry = toLogEntry('row_4', result);
    executionLog.push(logEntry);

    console.log('\n------------ Out-of-Order Patient ------------');
    console.log(JSON.stringify(logEntry, null, 2));
    console.log('----------------------------------------------\n');

    // Age 10 months + breaths 55 > 50 threshold for ≤12mo → FAST_BREATHING → CLINIC
    expect(result.errors).to.be.empty;
    expect(logEntry.final_outcome.triage).to.equal('clinic');
    expect(logEntry.final_outcome.reason).to.equal('fast_breathing');
  });

  it('missing field → throws error instead of defaulting', async () => {
    // Patient record missing chest_indrawing — should NOT silently default
    const patient = {
      child_name: 'Incomplete Child',
      age_months: 24,
      sex: 'male',
      convulsions_present: 'no',
      unusually_sleepy_or_unconscious: 'no',
      not_able_to_drink_or_feed_anything: 'no',
      vomits_everything: 'no',
      // chest_indrawing intentionally missing
      swelling_of_both_feet: 'no',
      muac_color: 'green',
      cough_present: 'no',
      diarrhoea_present: 'no',
      fever_present: 'no'
    };

    expect(() => translate(patient)).to.throw('Missing required field: chest_indrawing');
  });
});
