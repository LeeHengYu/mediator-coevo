# Task Instruction

## Task: Healthcare Cost-Benefit Analysis – 30-day vs 90-day Refill Cycle Margin Comparison

### Step 1: Inspect Input Files
Read and display the contents of all three input CSV files:
```
cat /root/acquisition_cost.csv
cat /root/packaging_cost.csv
cat /root/reimbursement.csv
```
Understand the column names, therapy names, and how they join together. The key join columns are `therapy` and `canister_size_units`.

### Step 2: Write a Python Script to Compute Everything
Create `/root/solve.py` with the following logic:

```python
import csv, json, math

# Read CSVs
def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))

acq = read_csv('/root/acquisition_cost.csv')
pkg = read_csv('/root/packaging_cost.csv')
reimb = read_csv('/root/reimbursement.csv')

# Build lookup dicts
# acquisition_cost.csv has: therapy, price_per_1000_doses_usd
acq_dict = {row['therapy']: float(row['price_per_1000_doses_usd']) for row in acq}

# packaging_cost.csv has: canister_size_units, packaging_cost_usd
# We need to match therapies to canister sizes. Check if packaging_cost.csv has a therapy column or just canister_size_units.
# If it only has canister_size_units, we need to get canister_size_units from acquisition_cost.csv or another file.
# Inspect carefully.

# reimbursement.csv has: therapy, reimbursement_per_fill_240_patients_usd (or similar)

patients = 240
fills_30 = 12
fills_90 = 4
doses_per_fill_30 = 60
doses_per_fill_90 = 180
threshold = 12000

# Build therapy list from acquisition_cost (the primary therapy list)
# For each therapy, find canister_size_units from acquisition_cost.csv
# Then look up packaging_cost_usd from packaging_cost.csv by canister_size_units
# Then look up reimbursement from reimbursement.csv by therapy

# IMPORTANT: Inspect the actual column names from each CSV. Adjust field names accordingly.
# Print the actual keys to debug:
print('acq keys:', acq[0].keys())
print('pkg keys:', pkg[0].keys())
print('reimb keys:', reimb[0].keys())
```

Run `python3 /root/solve.py` first just to see the column names, then refine.

### Step 3: Complete the Calculation Logic
After confirming column names, update solve.py with full logic:

For each therapy:
- `price_per_1000 = acq_dict[therapy]`
- `canister_size_units` from acquisition_cost.csv (the column should be there)
- `packaging_cost_usd` from packaging_cost.csv matched by `canister_size_units`
- `reimbursement_per_fill` from reimbursement.csv matched by `therapy`

Calculations per therapy:
- `annual_drug_cost_30 = (doses_per_fill_30 * fills_30 * patients) * price_per_1000 / 1000`
  - That is: total doses per year for all patients = 60 * 12 * 240 = 172,800 doses
  - Drug cost = 172800 * price_per_1000 / 1000
- `annual_drug_cost_90 = (doses_per_fill_90 * fills_90 * patients) * price_per_1000 / 1000`
  - Total doses = 180 * 4 * 240 = 172,800 doses (same!)
  - So drug costs should be identical for 30 and 90 day.
- `annual_packaging_cost_30 = packaging_cost_usd * patients * fills_30`
- `annual_packaging_cost_90 = packaging_cost_usd * patients * fills_90`
- `annual_reimbursement_30 = reimbursement_per_fill * fills_30`
- `annual_reimbursement_90 = reimbursement_per_fill * fills_90`
- `annual_margin = annual_reimbursement - annual_drug_cost - annual_packaging_cost`
- `difference = margin_90 - margin_30`

All currency values rounded to 2 decimals.

Sort therapies alphabetically by therapy name.

Compute totals by summing across therapies.

Decision rule:
- If `abs(total_difference) < 12000` → `adopt_90_day`
- Otherwise → `keep_30_day`

### Step 4: Generate `/root/cycle_margin_analysis.json`
Write the JSON with the exact schema specified. Use `json.dumps(obj, indent=2)`. Ensure:
- `assumptions` block has exact keys and values as specified
- `therapies` array sorted alphabetically
- `totals` block with all four fields
- `recommendation` with `decision` (exact slug) and `justification` (a brief string)

### Step 5: Generate `/root/cycle_margin_summary.md`
Write a markdown file with 4-8 non-empty lines including:
- Total 30-day margin (USD)
- Total 90-day margin (USD)
- Absolute difference (USD)
- Final decision using exact slug (`adopt_90_day` or `keep_30_day`)

Example format:
```
# Cycle Margin Summary

Total 30-day annual margin: $X.XX
Total 90-day annual margin: $Y.YY
Absolute margin difference: $Z.ZZ
Recommendation: adopt_90_day
```

### Step 6: Validate Outputs
1. Run `cat /root/cycle_margin_analysis.json` and verify:
   - Valid JSON (try `python3 -c "import json; json.load(open('/root/cycle_margin_analysis.json'))"`).
   - All required keys present at every level.
   - Therapies sorted alphabetically.
   - All currency values have exactly 2 decimal places.
   - Decision slug is exactly `adopt_90_day` or `keep_30_day`.
2. Run `cat /root/cycle_margin_summary.md` and verify:
   - 4-8 non-empty lines.
   - Contains total 30-day margin, total 90-day margin, absolute difference, and decision slug.

### Important Notes
- Read the CSV files carefully first. Column names may vary slightly from what's described.
- The reimbursement CSV says "per fill for 240 patients" — this means the value is already for all 240 patients combined per fill, NOT per patient per fill.
- The packaging cost is per patient per fill.
- Drug cost: price_per_1000_doses is the cost per 1000 doses. Total doses = doses_per_fill × fills_per_year × patients. Drug cost = total_doses × price_per_1000 / 1000.
- Double-check that drug costs are identical for 30-day and 90-day (since total annual doses are the same: 172,800 per therapy).
- Ensure the JSON numbers are actual numbers (not strings) rounded to 2 decimal places.

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