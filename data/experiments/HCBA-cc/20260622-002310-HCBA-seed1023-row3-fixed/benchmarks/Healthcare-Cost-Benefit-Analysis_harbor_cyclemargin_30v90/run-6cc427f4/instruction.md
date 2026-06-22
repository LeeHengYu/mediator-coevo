# Task Instruction

## Task: Healthcare Cost-Benefit Analysis — 30-day vs 90-day Refill Cycle Margin Comparison

### Step 1: Inspect Input Files
Read and display the contents of all three input CSV files:
```
cat /root/acquisition_cost.csv
cat /root/packaging_cost.csv
cat /root/reimbursement.csv
```
Understand the column names, therapy names, and how they join together. The key join columns are `therapy` and `canister_size_units`.

### Step 2: Write a Python Script to Perform the Analysis
Create `/root/solve.py` that does the following:

1. **Load CSVs** using the `csv` module (or `pandas` if available):
   - `acquisition_cost.csv` — contains at least `therapy`, `price_per_1000_doses_usd`
   - `packaging_cost.csv` — contains at least `canister_size_units`, `packaging_cost_usd`
   - `reimbursement.csv` — contains at least `therapy`, and a reimbursement column (the reimbursement per fill for 240 patients)

2. **Join data** by therapy. Match packaging cost by `canister_size_units` (acquisition_cost.csv should have a `canister_size_units` column that links to packaging_cost.csv).

3. **Constants**:
   - `patients_per_therapy = 240`
   - `doses_per_fill_30 = 60`, `fills_per_year_30 = 12`
   - `doses_per_fill_90 = 180`, `fills_per_year_90 = 4`
   - `switch_threshold_usd = 12000`

4. **Per-therapy calculations** (for each fill model, 30-day and 90-day):
   - `annual_drug_cost = (price_per_1000_doses_usd / 1000) * doses_per_fill * fills_per_year * patients_per_therapy`
   - `annual_packaging_cost = packaging_cost_usd * fills_per_year * patients_per_therapy`
   - `annual_reimbursement = reimbursement_per_fill_240_patients_usd * fills_per_year`
   - `annual_margin = annual_reimbursement - annual_drug_cost - annual_packaging_cost`
   - `annual_margin_difference_90_minus_30 = annual_margin_90_day - annual_margin_30_day`

5. **Totals**:
   - Sum all per-therapy `annual_margin_30_day_usd` → `total_annual_margin_30_day_usd`
   - Sum all per-therapy `annual_margin_90_day_usd` → `total_annual_margin_90_day_usd`
   - `total_annual_margin_difference_90_minus_30_usd = total_90 - total_30`
   - `absolute_total_margin_difference_usd = abs(total_annual_margin_difference_90_minus_30_usd)`

6. **Decision**:
   - If `absolute_total_margin_difference_usd < 12000` → `adopt_90_day`
   - Otherwise → `keep_30_day`
   - Write a short justification string.

7. **Round** all currency values to 2 decimal places.

8. **Sort** the therapies array alphabetically by `therapy` name.

9. **Output JSON** to `/root/cycle_margin_analysis.json` matching the exact schema from the instructions. Use `json.dump` with `indent=2`.

10. **Output Markdown** to `/root/cycle_margin_summary.md` with 4–8 non-empty lines including:
    - Total 30-day margin (USD)
    - Total 90-day margin (USD)
    - Absolute difference (USD)
    - Final decision using the exact slug `adopt_90_day` or `keep_30_day`

### Step 3: Run the Script
```
python3 /root/solve.py
```

### Step 4: Validate Outputs
1. **Check JSON validity and schema compliance**:
```
python3 -c "
import json
with open('/root/cycle_margin_analysis.json') as f:
    d = json.load(f)
assert 'assumptions' in d
assert 'therapies' in d and len(d['therapies']) > 0
assert 'totals' in d
assert 'recommendation' in d
assert d['recommendation']['decision'] in ('adopt_90_day', 'keep_30_day')
for t in d['therapies']:
    for k in ['therapy','price_per_1000_doses_usd','canister_size_units','packaging_cost_usd','reimbursement_per_fill_240_patients_usd','annual_drug_cost_30_day_usd','annual_drug_cost_90_day_usd','annual_packaging_cost_30_day_usd','annual_packaging_cost_90_day_usd','annual_reimbursement_30_day_usd','annual_reimbursement_90_day_usd','annual_margin_30_day_usd','annual_margin_90_day_usd','annual_margin_difference_90_minus_30_usd']:
        assert k in t, f'Missing key {k} in therapy {t}'
for k in ['total_annual_margin_30_day_usd','total_annual_margin_90_day_usd','total_annual_margin_difference_90_minus_30_usd','absolute_total_margin_difference_usd']:
    assert k in d['totals'], f'Missing key {k} in totals'
assert d['therapies'] == sorted(d['therapies'], key=lambda x: x['therapy']), 'Therapies not sorted alphabetically'
print('JSON schema validation passed')
print(json.dumps(d, indent=2))
"
```

2. **Check Markdown**:
```
cat /root/cycle_margin_summary.md
python3 -c "
with open('/root/cycle_margin_summary.md') as f:
    lines = [l.strip() for l in f if l.strip()]
print(f'Non-empty lines: {len(lines)}')
assert 4 <= len(lines) <= 8, f'Expected 4-8 non-empty lines, got {len(lines)}'
content = open('/root/cycle_margin_summary.md').read()
assert 'adopt_90_day' in content or 'keep_30_day' in content, 'Decision slug missing'
print('Markdown validation passed')
"
```

3. **Verify margin math for one therapy manually** — pick the first therapy from the JSON output and verify its annual_drug_cost, packaging_cost, reimbursement, and margin by hand-calculating from the raw CSV values. Print the comparison.

### Important Notes
- The reimbursement CSV gives reimbursement **per fill for 240 patients** — so annual reimbursement is simply `reimbursement_per_fill * fills_per_year` (do NOT multiply by patients again).
- Drug cost formula: `(price_per_1000_doses / 1000) * doses_per_fill * fills_per_year * patients` — this IS multiplied by patients because the price is per-dose.
- Packaging cost: `packaging_cost_usd * fills_per_year * patients` — per patient per fill.
- Be careful about the join between acquisition_cost and packaging_cost — match on `canister_size_units`.
- Read the actual column names from the CSVs before coding; adapt column name references to match exactly what's in the files.

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