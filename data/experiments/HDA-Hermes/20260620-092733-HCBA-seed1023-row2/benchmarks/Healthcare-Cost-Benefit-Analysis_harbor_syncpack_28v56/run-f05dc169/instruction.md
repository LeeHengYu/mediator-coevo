# Task Instruction

## Task: Healthcare Cost-Benefit Analysis — 28-day vs 56-day Syncpack

### Step 1: Inspect Input Files
Read and display the contents of all three input CSV files:
```
cat /root/ingredient_cost.csv
cat /root/card_cost.csv
cat /root/reimbursement.csv
```
Understand the columns, medications listed, and how they join together (likely by medication name or blister_card_count).

### Step 2: Write a Python Script to Perform the Analysis
Create `/root/solve.py` that does the following:

1. **Load CSVs** using the `csv` module (or `pandas` if available).
2. **Join the three datasets** by medication name. The `card_cost.csv` is matched to each medication by its `blister_card_count` field (each medication has a blister_card_count in one of the files; use that to look up the card_cost_usd).
3. **For each medication, compute:**
   - `annual_drug_cost_28_day_usd = (price_per_1000_capsules_usd / 1000) * 56 * 12 * 180`
   - `annual_drug_cost_56_day_usd = (price_per_1000_capsules_usd / 1000) * 112 * 6 * 180`
     - Note: both should be identical since 56*12 == 112*6 == 672 capsules/patient/year * 180 patients. But compute them separately per the schema.
   - `annual_packaging_cost_28_day_usd = card_cost_usd * 180 * 12` (one card per patient per fill)
   - `annual_packaging_cost_56_day_usd = card_cost_usd * 180 * 6` (one card per patient per fill, matched by blister_card_count for the 56-day model)
     - **IMPORTANT**: The blister_card_count may differ between 28-day and 56-day models. For 28-day, each fill is 56 capsules, so the blister card holds 56. For 56-day, each fill is 112 capsules, so the blister card holds 112. Look up the card_cost_usd from `card_cost.csv` using the appropriate blister_card_count for each model. If `card_cost.csv` has rows keyed by `blister_card_count`, use count=56 for 28-day model and count=112 for 56-day model (or whatever counts match the capsules_per_fill values). Inspect the actual CSV to confirm.
   - `annual_reimbursement_28_day_usd = reimbursement_per_cycle_180_patients_usd * 12`
   - `annual_reimbursement_56_day_usd = reimbursement_per_cycle_180_patients_usd * 6`
   - `annual_margin_28_day_usd = annual_reimbursement_28_day_usd - annual_drug_cost_28_day_usd - annual_packaging_cost_28_day_usd`
   - `annual_margin_56_day_usd = annual_reimbursement_56_day_usd - annual_drug_cost_56_day_usd - annual_packaging_cost_56_day_usd`
   - `annual_margin_difference_56_minus_28_usd = annual_margin_56_day_usd - annual_margin_28_day_usd`
   - Round all currency values to 2 decimal places.

4. **Sort medications alphabetically** by medication name (case-insensitive sort, but preserve original casing).

5. **Compute totals:**
   - `total_annual_margin_28_day_usd` = sum of all medications' `annual_margin_28_day_usd`
   - `total_annual_margin_56_day_usd` = sum of all medications' `annual_margin_56_day_usd`
   - `total_annual_margin_difference_56_minus_28_usd` = sum of all medications' `annual_margin_difference_56_minus_28_usd`
   - `absolute_total_margin_difference_usd` = abs(total_annual_margin_difference_56_minus_28_usd)
   - Round all to 2 decimals.

6. **Decision rule:**
   - If `absolute_total_margin_difference_usd < 9000`, decision = `"convert_to_56_day"`
   - Otherwise, decision = `"keep_28_day"`
   - Write a justification string that references the absolute difference and threshold.

7. **Write `/root/syncpack_analysis.json`** with the exact schema specified. Use `json.dumps` with `indent=2`. Ensure all numeric values are floats rounded to 2 decimals (not strings).

8. **Write `/root/syncpack_summary.md`** with 4-8 non-empty lines containing:
   - Total 28-day margin (USD) with the dollar amount
   - Total 56-day margin (USD) with the dollar amount
   - Absolute difference (USD) with the dollar amount
   - Final decision using the exact slug `convert_to_56_day` or `keep_28_day`

### Step 3: Run the Script
```
python3 /root/solve.py
```

### Step 4: Validate Outputs
1. Verify JSON is valid:
```
python3 -c "import json; d=json.load(open('/root/syncpack_analysis.json')); print(json.dumps(d, indent=2))"
```
2. Check all required keys exist in the JSON:
```
python3 -c "
import json
d = json.load(open('/root/syncpack_analysis.json'))
assert 'assumptions' in d
assert 'medications' in d and len(d['medications']) > 0
assert 'totals' in d
assert 'recommendation' in d
for m in d['medications']:
    for k in ['medication','price_per_1000_capsules_usd','blister_card_count','card_cost_usd','reimbursement_per_cycle_180_patients_usd','annual_drug_cost_28_day_usd','annual_drug_cost_56_day_usd','annual_packaging_cost_28_day_usd','annual_packaging_cost_56_day_usd','annual_reimbursement_28_day_usd','annual_reimbursement_56_day_usd','annual_margin_28_day_usd','annual_margin_56_day_usd','annual_margin_difference_56_minus_28_usd']:
        assert k in m, f'Missing key {k} in medication {m.get("medication","?")}'  
print('All keys present')
meds = [m['medication'] for m in d['medications']]
assert meds == sorted(meds, key=str.lower), 'Medications not sorted alphabetically'
print('Sort order correct')
print('Decision:', d['recommendation']['decision'])
"
```
3. Check the markdown summary:
```
cat /root/syncpack_summary.md
wc -l /root/syncpack_summary.md
```
Verify it has 4-8 non-empty lines and includes the required information.

### Key Pitfalls to Avoid
- **Packaging cost lookup**: The card_cost_usd likely differs between 28-day (56-capsule card) and 56-day (112-capsule card) models. Use the correct blister_card_count to look up the right cost for each model. Inspect `card_cost.csv` carefully.
- **Reimbursement**: The reimbursement CSV gives per-cycle reimbursement for 180 patients. Multiply by fills_per_year for each model.
- **Rounding**: Round ALL currency outputs to exactly 2 decimal places.
- **Sort**: Sort medications array alphabetically by medication name.
- **Decision slug**: Use exactly `convert_to_56_day` or `keep_28_day` — no variations.

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