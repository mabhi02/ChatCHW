const { expect } = require('chai');
const Harness = require('cht-conf-test-harness');

const harness = new Harness({ verbose: true });

describe('Debug flat array', () => {
  before(async () => { return await harness.start(); });
  after(async () => { return await harness.stop(); });
  beforeEach(async () => { return await harness.clear(); });

  it('flat array all answers as one page', async () => {
    const r = await harness.fillForm(
      'orchestrator',
      ['Test', 24, 'male', 'no', 'no', 'no', 'no', 'no', 'no', 'green', 'no', 'no', 'no']
    );
    console.log('\n========== ERRORS ==========');
    console.log(JSON.stringify(r.errors, null, 2));
    if (r.report) {
      console.log('\n========== FIELDS ==========');
      console.log(JSON.stringify(r.report.fields, null, 2));
    }
    console.log('========== END ==========\n');
  });
});