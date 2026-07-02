# Task Instruction

Execute the following steps in order:

1. **Inspect the input files** to understand their structure:
```bash
cat /root/ingredient_cost.csv
cat /root/card_cost.csv
cat /root/reimbursement.csv
```

2. **Inspect the test file** to understand exact validation expectations:
```bash
cat /root/test_output.py
```

3. **Create a Python script** `/root/solve.py` that:
   - Reads the three CSV files using the `csv` module.
   - For each medication (joined across the three CSVs by medication name):
     - Extracts `price_per_1000_capsules_usd`, `blister_card_count`, `card_cost_usd`, and `reimbursement_per_cycle_180_patients_usd`.
     - Computes:
       - `annual_drug_cost_28_day_usd = (price_per_1000_capsules_usd / 1000) * 56 * 12 * 180`
       - `annual_drug_cost_56_day_usd = (price_per_1000_capsules_usd / 1000) * 112 * 6 * 180`
       - `annual_packaging_cost_28_day_usd = card_cost_usd * 180 * 12`
       - `annual_packaging_cost_56_day_usd = card_cost_usd * 180 * 6`
       - `annual_reimbursement_28_day_usd = reimbursement_per_cycle_180_patients_usd * 12`
       - `annual_reimbursement_56_day_usd = reimbursement_per_cycle_180_patients_usd * 6`
       - `annual_margin_28_day_usd = annual_reimbursement_28_day_usd - annual_drug_cost_28_day_usd - annual_packaging_cost_28_day_usd`
       - `annual_margin_56_day_usd = annual_reimbursement_56_day_usd - annual_drug_cost_56_day_usd - annual_packaging_cost_56_day_usd`
       - `annual_margin_difference_56_minus_28_usd = annual_margin_56_day_usd - annual_margin_28_day_usd`
     - All currency values rounded to 2 decimal places.
   - Sorts the medications list alphabetically by `medication` name.
   - Computes totals:
     - `total_annual_margin_28_day_usd` = sum of all `annual_margin_28_day_usd`
     - `total_annual_margin_56_day_usd` = sum of all `annual_margin_56_day_usd`
     - `total_annual_margin_difference_56_minus_28_usd` = sum of all `annual_margin_difference_56_minus_28_usd`
     - `absolute_total_margin_difference_usd` = abs(total_annual_margin_difference_56_minus_28_usd)
     - All rounded to 2 decimals.
   - Decision rule: if `absolute_total_margin_difference_usd < 9000`, decision is `"convert_to_56_day"`, else `"keep_28_day"`.
   - Writes `/root/syncpack_analysis.json` with **exactly** these top-level keys: `assumptions`, `medications`, `totals`, `recommendation`.
   - The `assumptions` object has exactly: `patients_per_medication`, `fills_per_year_28_day`, `fills_per_year_56_day`, `capsules_per_fill_28_day`, `capsules_per_fill_56_day`, `switch_threshold_usd`.
   - Each medication object has **exactly** these keys (no more, no fewer): `medication`, `price_per_1000_capsules_usd`, `blister_card_count`, `card_cost_usd`, `reimbursement_per_cycle_180_patients_usd`, `annual_drug_cost_28_day_usd`, `annual_drug_cost_56_day_usd`, `annual_packaging_cost_28_day_usd`, `annual_packaging_cost_56_day_usd`, `annual_reimbursement_28_day_usd`, `annual_reimbursement_56_day_usd`, `annual_margin_28_day_usd`, `annual_margin_56_day_usd`, `annual_margin_difference_56_minus_28_usd`.
   - The `totals` object has **exactly**: `total_annual_margin_28_day_usd`, `total_annual_margin_56_day_usd`, `total_annual_margin_difference_56_minus_28_usd`, `absolute_total_margin_difference_usd`.
   - The `recommendation` object has **exactly**: `decision`, `justification`.
   - Writes `/root/syncpack_summary.md` with 4-8 non-empty lines including: total 28-day margin (USD), total 56-day margin (USD), absolute difference (USD), and the exact decision slug (`convert_to_56_day` or `keep_28_day`).

4. **CRITICAL KEY NAMING**: Every key in the JSON must use the `_usd` suffix as shown in the schema. The previous failure was caused by keys like `annual_packaging_cost_28_day` instead of `annual_packaging_cost_28_day_usd`. Double-check every key name matches the schema exactly.

5. **Run the solver**:
```bash
cd /root && python solve.py
```

6. **Validate the output** by inspecting the JSON keys:
```bash
python3 -c "
import json
with open('/root/syncpack_analysis.json') as f:
    data = json.load(f)
print('Top keys:', sorted(data.keys()))
print('Assumptions keys:', sorted(data['assumptions'].keys()))
if data['medications']:
    print('Med keys:', sorted(data['medications'][0].keys()))
    print('Med count:', len(data['medications']))
    print('First med:', data['medications'][0]['medication'])
print('Totals keys:', sorted(data['totals'].keys()))
print('Totals:', data['totals'])
print('Recommendation:', data['recommendation'])
"
```

7. **Verify the summary file**:
```bash
cat /root/syncpack_summary.md
```

8. **Run the test suite**:
```bash
cd /root && python -m pytest test_output.py -v
```

9. If any test fails, read the error carefully, fix the issue in `solve.py`, re-run the solver, and re-run the tests. Pay special attention to:
   - Extra or missing keys in any JSON object
   - Key naming mismatches (especially missing `_usd` suffixes)
   - Sorting order of medications
   - Rounding precision
   - Summary file content requirements

# Executor Policy

---
name: executor
description: Portable executor policy for workflow, verification, resource use, and failure handling across task runtimes.
---

## Executor Policy

Use this skill as execution policy, not as domain-specific task knowledge. When
task-local curated skills or resources are available, prefer them for domain
details and use this policy for workflow control.

## Task Execution

1. Read the task instruction, task resources, and verifier contract before editing.
2. Identify the scoring mechanism and the smallest command that can reproduce the
   failure or verify the expected behavior.
3. Inspect existing files and task-local resources before making changes.
4. Make the smallest source change that satisfies the task and verifier contract.
5. Keep a compact record of the concrete evidence behind the change: observed
   failure, files inspected, edit made, and verifier result.
6. Run targeted verification before broad verification when practical.

## File Editing

1. Read the actual current file contents immediately before making any edit.
   Never rely on memory, prior snapshots, or assumed content.
2. Prefer direct in-place edits over patch or diff application when the exact
   current context is uncertain.
3. If using a patch or diff, confirm that every context line exists verbatim in
   the file before applying it.
4. If a patch hunk fails to apply, re-read the affected file region and perform
   the edit directly instead of retrying the same patch.
5. After any edit, re-read the affected region to confirm the change landed.

## Build and Test Fixes

When a task requires fixing a broken build, failing test, or generated artifact:

1. Run the relevant build, test, or verifier command first to capture the
   baseline failure.
2. Identify the specific error message, file, line, or expected output before
   editing.
3. Apply the smallest fix, then re-run the same targeted command.
4. Treat newly introduced failures as separate sub-tasks and resolve them in
   order.
5. Do not mark the task complete until the verifier-relevant command succeeds or
   the remaining failure is clearly outside the task boundary.

## Artifact-Contract Handling

Do not treat artifacts as ordinary text files. Treat them as contract-bearing
interfaces between input data, generated output, verifier checks, and downstream
consumers.

When a task requires reading, modifying, or generating an artifact such as JSON,
DOT, reports, configs, generated source, schemas, datasets, or parsed outputs:

1. Identify the artifact contract first: format, schema, required fields,
   identifiers, references, ordering, examples, verifier assertions, and
   consuming code.
2. Inspect representative source artifacts directly before deciding how to
   transform or preserve them.
3. Determine whether the task calls for preservation, transformation, repair,
   generation, or validation.
4. Preserve required literals, identifiers, references, ordering, and
   representative content unless the contract explicitly requires a change.
5. Do not invent, drop, rename, normalize, collapse, expand, or repair artifact
   elements unless the verifier or consumer contract requires that behavior.
6. Prefer structured parsers, serializers, validators, or existing consumer code
   over ad hoc string manipulation when they are available.
7. After producing the artifact, run targeted checks for parseability, required
   keys or IDs, reference consistency, expected counts, preserved content, and
   format-specific validity.
8. If targeted checks regress or become unusable after a change, stop expanding
   the solution. Re-inspect the source contract and narrow the edit before trying
   a broader repair.

A plausible-looking artifact is not sufficient evidence. The artifact is only
correct when it satisfies the task contract under the verifier or consuming
code.

## Constraints

- Do not bypass, remove, or weaken tests, verifier scripts, fixtures, or expected
  output checks.
- Do not treat this policy as overriding task-specific instructions or verifier
  requirements.
- On tool or environment errors, retry once when the retry is safe, then report
  the failure with the command and error output.
- On ambiguous instructions, make a conservative assumption and continue.

# Task Resources

Inspect the task files, environment, tests, and expected outputs directly.

# Verifier Contract

Success is judged by the SkillFlow verifier for this task.
Do not bypass, remove, or weaken verifier scripts, tests, fixtures, or expected-output checks.
Run the provided tests or verifier command when practical before finalizing.
Task metadata: author_email=gpt54@example.com, author_name=GPT-5.4, category=financial-analysis, difficulty=medium, tags=[med-sync, packaging, csv, json, decision-analysis].
Verifier config: timeout_sec=900.0.