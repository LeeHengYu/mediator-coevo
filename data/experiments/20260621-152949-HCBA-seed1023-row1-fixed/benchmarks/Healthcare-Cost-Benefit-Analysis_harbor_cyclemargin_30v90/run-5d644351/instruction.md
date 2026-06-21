# Task Instruction

Execute the following steps in order:

1. **Inspect the input files** to understand their structure and contents:
```bash
cat /root/acquisition_cost.csv
cat /root/packaging_cost.csv
cat /root/reimbursement.csv
```

2. **Inspect the test file** to understand exact verifier expectations:
```bash
cat /root/tests/test_outputs.py
```

3. **Create `/root/solve.py`** — a Python script that:
   - Reads all three CSV files using the `csv` module.
   - Joins them by therapy name. The `packaging_cost.csv` should be matched by `canister_size_units` (check the actual column names after inspecting the files — it may need to be joined on therapy name or canister size; inspect carefully).
   - For each therapy, computes:
     - `annual_drug_cost = (price_per_1000_doses_usd / 1000) * doses_per_fill * fills_per_year * 240`
       - 30-day: doses_per_fill=60, fills_per_year=12
       - 90-day: doses_per_fill=180, fills_per_year=4
       - NOTE: annual_drug_cost should be identical for 30-day and 90-day since 60*12 == 180*4 == 720 doses/patient/year. But compute them separately per the schema.
     - `annual_packaging_cost = packaging_cost_usd * 240 * fills_per_year`
       - 30-day: fills_per_year=12
       - 90-day: fills_per_year=4
     - `annual_reimbursement = reimbursement_per_fill_240_patients_usd * fills_per_year`
       - 30-day: fills_per_year=12
       - 90-day: fills_per_year=4
     - `annual_margin = annual_reimbursement - annual_drug_cost - annual_packaging_cost`
     - `annual_margin_difference_90_minus_30 = annual_margin_90_day - annual_margin_30_day`
   - All currency values rounded to 2 decimal places.
   - Sorts therapies alphabetically by `therapy` name.
   - Computes totals by summing across all therapies.
   - Applies decision rule: if `abs(total_difference) < 12000` → `adopt_90_day`, else `keep_30_day`.
   - Writes `/root/cycle_margin_analysis.json` with **exactly** these top-level keys: `assumptions`, `therapies`, `totals`, `recommendation`.
     - `assumptions` contains the fixed parameters.
     - `recommendation` contains `decision` and `justification`.
   - Writes `/root/cycle_margin_summary.md` with 4-8 non-empty lines including total 30-day margin, total 90-day margin, absolute difference, and the exact decision slug.

4. **Run the script:**
```bash
cd /root && python solve.py
```

5. **Validate the JSON output structure:**
```bash
python3 -c "
import json
with open('/root/cycle_margin_analysis.json') as f:
    d = json.load(f)
assert set(d.keys()) == {'assumptions', 'therapies', 'totals', 'recommendation'}, f'Top keys: {set(d.keys())}'
assert 'decision' in d['recommendation'], 'Missing decision in recommendation'
assert 'justification' in d['recommendation'], 'Missing justification in recommendation'
assert isinstance(d['therapies'], list) and len(d['therapies']) > 0, 'Empty therapies'
for t in d['therapies']:
    for k in ['therapy','price_per_1000_doses_usd','canister_size_units','packaging_cost_usd','reimbursement_per_fill_240_patients_usd','annual_drug_cost_30_day_usd','annual_drug_cost_90_day_usd','annual_packaging_cost_30_day_usd','annual_packaging_cost_90_day_usd','annual_reimbursement_30_day_usd','annual_reimbursement_90_day_usd','annual_margin_30_day_usd','annual_margin_90_day_usd','annual_margin_difference_90_minus_30_usd']:
        assert k in t, f'Missing key {k} in therapy {t.get("therapy","?")}'  
for k in ['total_annual_margin_30_day_usd','total_annual_margin_90_day_usd','total_annual_margin_difference_90_minus_30_usd','absolute_total_margin_difference_usd']:
    assert k in d['totals'], f'Missing key {k} in totals'
print('JSON structure OK')
print(json.dumps(d, indent=2))
"
```

6. **Validate the markdown:**
```bash
python3 -c "
lines = [l for l in open('/root/cycle_margin_summary.md').read().strip().split('\n') if l.strip()]
assert 4 <= len(lines) <= 8, f'Line count: {len(lines)}'
content = open('/root/cycle_margin_summary.md').read()
assert 'adopt_90_day' in content or 'keep_30_day' in content, 'Missing decision slug'
print('Markdown OK')
print(content)
"
```

7. **Run the test suite:**
```bash
cd /root && python -m pytest tests/test_outputs.py -v
```

8. If any test fails, read the error carefully, re-inspect the relevant input files and computations, fix `solve.py`, re-run it, and re-run the tests. Pay special attention to:
   - How packaging cost is matched to therapies (by therapy name? by canister_size_units?).
   - Whether reimbursement is already for 240 patients per fill (as stated) — do NOT multiply by 240 again.
   - Numerical precision — round each value to 2 decimals.
   - The exact JSON key names and nesting structure.

IMPORTANT NOTES from previous failure:
- The `decision` and `justification` MUST be nested inside a `recommendation` object, NOT at the top level.
- The `assumptions` key MUST be present at the top level of the JSON.
- From the avoid artifact: be very careful with the margin calculation formula. Do not multiply reimbursement by number of patients if it's already per-fill for 240 patients.

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
Task metadata: author_email=gpt54@example.com, author_name=GPT-5.4, category=financial-analysis, difficulty=medium, tags=[healthcare, unit-economics, csv, json, decision-analysis].
Verifier config: timeout_sec=900.0.