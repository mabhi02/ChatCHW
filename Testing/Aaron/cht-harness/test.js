// Orchestrator Harness Prototype
//
// Loads the orchestrator XLSForm via the CHT test harness, runs patients
// through the full form, and captures the output in Emett's execution log
// schema for cross-engine comparison.
//
// Context: the /api/v2/records endpoint on our local CHT server can't find
// the orchestrator form ("Form not found: ORCHESTRATOR"), and the DMN approach
// requires strictly normalized values, so this is an alternative.

const { expect } = require('chai');
const Harness = require('cht-conf-test-harness');
const fs = require('fs');

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


  // Answer structure reference (max 19 questions as flat array):
  //  1. child_name (string)
  //  2. age_months (int 0-59, must be JS number not string)
  //  3. sex (male/female)
  //  4-9. danger signs: convulsions, sleepy, not_drinking, vomits,
  //       chest_indrawing, swelling (yes/no)
  //  10. muac_color (red/yellow/green)
  //  11. cough_present (yes/no), if yes: +cough_duration_days, +breaths_per_minute
  //  next: diarrhoea_present (yes/no), if yes: +diarrhoea_duration_days, +blood_in_stool
  //  next: fever_present (yes/no), if yes: +fever_duration_days, +rdt_result


  it('healthy patient → home', async () => {
    const result = await harness.fillForm('orchestrator', [
      'Test Child', 24, 'male',
      'no', 'no', 'no', 'no', 'no', 'no', 'green',
      'no',
      'no',
      'no'
    ]);

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
    const result = await harness.fillForm('orchestrator', [
      'Sick Child', 18, 'female',
      'no', 'no', 'no', 'no', 'yes', 'no', 'green',
      'no',
      'no',
      'no'
    ]);

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
    const result = await harness.fillForm('orchestrator', [
      'Cough Child', 24, 'male',
      'no', 'no', 'no', 'no', 'no', 'no', 'green',
      'yes', 3, 45,
      'no',
      'no'
    ]);

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
});
