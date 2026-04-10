// Orchestrator Harness Prototype
//
// Loads the orchestrator XLSForm via the CHT test harness, runs one patient
// through it, and captures the output in our main loop's log format.
//
// Context: the /api/v2/records endpoint on our local CHT server can't find 
// the orchestrator form ("Form not found: ORCHESTRATOR"), and the DMN approach 
// requires strictly normalized values, so this is an alternative.

const { expect } = require('chai');
const Harness = require('cht-conf-test-harness');
const fs = require('fs');

const harness = new Harness({ verbose: true });

// Converts the harness result into the log format used by main_loop.py.
// On error, returns a missing data alert entry. On success, returns a full
// triage entry with the fields computed by the orchestrator form.
function toLogEntry(patientRow, age, complaint, harnessResult) {
  if (harnessResult.errors && harnessResult.errors.length > 0) {
    return {
      patient_row: patientRow,
      alert: `Missing Data — ${JSON.stringify(harnessResult.errors)}. Patient skipped.`
    };
  }

  const fields = harnessResult.report.fields;
  return {
    patient_row: patientRow,
    age: age,
    complaint: complaint,
    final_triage: fields.final_triage,
    triage_reason_code: fields.triage_reason_code,
    final_advice: fields.final_advice
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

  it('runs one patient through the orchestrator form and captures log output', async () => {
    const testPatient = {
      row: 1,
      age: 24,
      complaint: 'Routine check',
      answers: [
        'Test Child', 24, 'male',
        'no', 'no', 'no', 'no', 'no', 'no', 'green',
        'no',
        'no',
        'no'
      ]
    };

    const result = await harness.fillForm('orchestrator', testPatient.answers);
    const logEntry = toLogEntry(testPatient.row, testPatient.age, testPatient.complaint, result);
    executionLog.push(logEntry);

    console.log('\n------------ Log Entry ------------');
    console.log(JSON.stringify(logEntry, null, 2));
    console.log('-----------------------------------\n');

    expect(result.errors).to.be.empty;
    expect(logEntry).to.have.property('final_triage');
    expect(logEntry).to.have.property('triage_reason_code');
    expect(logEntry).to.have.property('final_advice');
  });

  it('runs a sick patient and confirms triage changes based on symptoms', async () => {
    const sickPatient = {
      row: 2,
      age: 18,
      complaint: 'Difficulty breathing',
      answers: [
        'Sick Child', 18, 'female',
        'no', 'no', 'no', 'no', 'yes', 'no', 'green',
        'no',
        'no',
        'no'
      ]
    };

    const result = await harness.fillForm('orchestrator', sickPatient.answers);
    const logEntry = toLogEntry(sickPatient.row, sickPatient.age, sickPatient.complaint, result);
    executionLog.push(logEntry);

    console.log('\n------ Sick Patient Log Entry ------');
    console.log(JSON.stringify(logEntry, null, 2));
    console.log('------------------------------------\n');

    expect(result.errors).to.be.empty;
    expect(logEntry.final_triage).to.equal('HOSPITAL');
  });
});
