# Task Instruction

Execute the following steps in order:

1. **Inspect the input files** to understand their structure and column names:
```bash
cat /root/acquisition_cost.csv
cat /root/packaging_cost.csv
cat /root/reimbursement.csv
```

2. **Inspect the test file** to understand exact validation expectations:
```bash
cat /root/tests/test_output*.py 2>/dev/null || cat /root/test_output*.py 2>/dev/null || find /root -name 'test_*.py' -exec cat {} \;
```

3. **Write and run a Python script** `/root/solve.py` that:

   a. Reads the three CSV files using the `csv` module.
   b. Joins them by therapy name (inspect the CSVs first to confirm the join key column name — likely `therapy`).
   c. For each therapy, computes:
      - `annual_drug_cost_30_day_usd = (price_per_1000_doses_usd / 1000) * 60 * 12 * 240`
      - `annual_drug_cost_90_day_usd = (price_per_1000_doses_usd / 1000) * 180 * 4 * 240`
        (Note: both should be identical since 60*12 == 180*4 == 720 doses/patient/year * 240 patients)
      - `annual_packaging_cost_30_day_usd = packaging_cost_usd * 240 * 12`
      - `annual_packaging_cost_90_day_usd = packaging_cost_usd * 240 * 4`
      - `annual_reimbursement_30_day_usd = reimbursement_per_fill_240_patients_usd * 12`
      - `annual_reimbursement_90_day_usd = reimbursement_per_fill_240_patients_usd * 4`
      - `annual_margin_30_day_usd = annual_reimbursement_30_day - annual_drug_cost_30_day - annual_packaging_cost_30_day`
      - `annual_margin_90_day_usd = annual_reimbursement_90_day - annual_drug_cost_90_day - annual_packaging_cost_90_day`
      - `annual_margin_difference_90_minus_30_usd = annual_margin_90_day_usd - annual_margin_30_day_usd`
   d. All currency values rounded to 2 decimal places.
   e. Sorts therapies alphabetically by `therapy`.
   f. Computes totals:
      - `total_annual_margin_30_day_usd` = sum of all `annual_margin_30_day_usd`
      - `total_annual_margin_90_day_usd` = sum of all `annual_margin_90_day_usd`
      - `total_annual_margin_difference_90_minus_30_usd` = sum of all `annual_margin_difference_90_minus_30_usd`
      - `absolute_total_margin_difference_usd` = abs(total_annual_margin_difference_90_minus_30_usd)
   g. Decision rule: if `absolute_total_margin_difference_usd < 12000`, decision = `adopt_90_day`; otherwise `keep_30_day`.
   h. Writes `/root/cycle_margin_analysis.json` with **exactly** this nested structure (no extra keys at root):
      ```json
      {
        "assumptions": {
          "patients_per_therapy": 240,
          "fills_per_year_30_day": 12,
          "fills_per_year_90_day": 4,
          "doses_per_fill_30_day": 60,
          "doses_per_fill_90_day": 180,
          "switch_threshold_usd": 12000
        },
        "therapies": [ ... sorted therapy objects with ALL fields listed in schema ... ],
        "totals": { ... },
        "recommendation": {
          "decision": "adopt_90_day" or "keep_30_day",
          "justification": "<a short sentence explaining the decision>"
        }
      }
      ```
      Each therapy object must include: `therapy`, `price_per_1000_doses_usd`, `canister_size_units`, `packaging_cost_usd`, `reimbursement_per_fill_240_patients_usd`, `annual_drug_cost_30_day_usd`, `annual_drug_cost_90_day_usd`, `annual_packaging_cost_30_day_usd`, `annual_packaging_cost_90_day_usd`, `annual_reimbursement_30_day_usd`, `annual_reimbursement_90_day_usd`, `annual_margin_30_day_usd`, `annual_margin_90_day_usd`, `annual_margin_difference_90_minus_30_usd`.
   i. Writes `/root/cycle_margin_summary.md` with 4–8 non-empty lines including:
      - Total 30-day margin in USD
      - Total 90-day margin in USD
      - Absolute difference in USD
      - The exact decision slug (`adopt_90_day` or `keep_30_day`)

4. **Run the script**:
```bash
cd /root && python solve.py
```

5. **Validate the output JSON** — confirm it has exactly the 4 root keys, therapies are sorted, and all fields present:
```bash
python3 -c "
import json
with open('/root/cycle_margin_analysis.json') as f:
    d = json.load(f)
print('Root keys:', sorted(d.keys()))
assert sorted(d.keys()) == ['assumptions', 'recommendation', 'therapies', 'totals'], 'Root key mismatch!'
for t in d['therapies']:
    required = ['therapy','price_per_1000_doses_usd','canister_size_units','packaging_cost_usd','reimbursement_per_fill_240_patients_usd','annual_drug_cost_30_day_usd','annual_drug_cost_90_day_usd','annual_packaging_cost_30_day_usd','annual_packaging_cost_90_day_usd','annual_reimbursement_30_day_usd','annual_reimbursement_90_day_usd','annual_margin_30_day_usd','annual_margin_90_day_usd','annual_margin_difference_90_minus_30_usd']
    for r in required:
        assert r in t, f'Missing {r} in therapy {t.get(\"therapy\",\"?\")}'  
print('Therapies:', [t['therapy'] for t in d['therapies']])
assert d['therapies'] == sorted(d['therapies'], key=lambda x: x['therapy']), 'Not sorted!'
print('Totals:', d['totals'])
print('Recommendation:', d['recommendation'])
print('ALL CHECKS PASSED')
"
```

6. **Validate the summary markdown**:
```bash
python3 -c "
with open('/root/cycle_margin_summary.md') as f:
    lines = [l.strip() for l in f if l.strip()]
print(f'Non-empty lines: {len(lines)}')
assert 4 <= len(lines) <= 8, f'Expected 4-8 lines, got {len(lines)}'
content = ' '.join(lines).lower()
assert 'adopt_90_day' in content or 'keep_30_day' in content, 'Missing decision slug'
print('Summary OK')
for l in lines: print(l)
"
```

7. **Run the test suite** if present:
```bash
cd /root && python -m pytest tests/ -v 2>/dev/null || python -m pytest . -v -k test_output 2>/dev/null || echo 'No test suite found'
```

IMPORTANT NOTES:
- The `canister_size_units` field in each therapy object should be an integer (from the CSV), not a float.
- The `packaging_cost_usd` in the packaging CSV is the cost per patient per fill. Match it to the therapy via `canister_size_units`.
- The reimbursement CSV gives reimbursement per fill for 240 patients (the entire cohort), not per patient.
- Do NOT add any extra keys at the root level of the JSON. Only `assumptions`, `therapies`, `totals`, `recommendation`.
- Read the CSV column headers carefully — use the exact column names from the files for lookups.

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