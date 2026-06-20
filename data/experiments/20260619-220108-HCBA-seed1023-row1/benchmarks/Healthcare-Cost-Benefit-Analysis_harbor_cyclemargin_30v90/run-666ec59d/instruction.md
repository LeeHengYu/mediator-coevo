# Task Instruction

## Task: Healthcare Cost-Benefit Analysis — 30-day vs 90-day Fill Cycle Margin Comparison

### Step 1: Inspect Input Files
Read and display the contents of all three input CSV files:
```
cat /root/acquisition_cost.csv
cat /root/packaging_cost.csv
cat /root/reimbursement.csv
```
Understand the columns, therapy names, and how they join together. The key join logic:
- `acquisition_cost.csv` has `therapy`, `price_per_1000_doses_usd` (and likely `canister_size_units`)
- `packaging_cost.csv` has `canister_size_units` and `packaging_cost_usd` — join to acquisition data via `canister_size_units`
- `reimbursement.csv` has `therapy` and reimbursement per fill for 240 patients

### Step 2: Write a Python Script
Create and run a Python script `/root/solve.py` that does the following:

```python
import csv, json, math

# 1. Read acquisition_cost.csv
# 2. Read packaging_cost.csv
# 3. Read reimbursement.csv
# 4. For each therapy, merge data by matching:
#    - acquisition_cost and reimbursement on 'therapy'
#    - acquisition_cost and packaging_cost on 'canister_size_units'
# 5. Compute per-therapy values:

# Constants
patients = 240
fills_30 = 12
fills_90 = 4
doses_per_fill_30 = 60
doses_per_fill_90 = 180

# For each therapy:
#   annual_drug_cost = (price_per_1000_doses_usd / 1000) * doses_per_fill * fills_per_year * patients
#     - 30-day: doses_per_fill=60, fills=12
#     - 90-day: doses_per_fill=180, fills=4
#   NOTE: 60*12 = 720 doses/patient/year, 180*4 = 720 doses/patient/year — drug cost should be same!
#
#   annual_packaging_cost = packaging_cost_usd * patients * fills_per_year
#     - 30-day: fills=12
#     - 90-day: fills=4
#
#   annual_reimbursement = reimbursement_per_fill_240_patients * fills_per_year
#     - 30-day: fills=12
#     - 90-day: fills=4
#
#   annual_margin = annual_reimbursement - annual_drug_cost - annual_packaging_cost
#   difference = margin_90 - margin_30

# 6. Totals: sum all per-therapy margins and differences
# 7. Decision: if abs(total_difference) < 12000 -> 'adopt_90_day', else 'keep_30_day'
# 8. Round all currency to 2 decimals
# 9. Sort therapies alphabetically by 'therapy'
# 10. Write /root/cycle_margin_analysis.json with exact schema
# 11. Write /root/cycle_margin_summary.md (4-8 non-empty lines with required info)
```

IMPORTANT details:
- `packaging_cost_usd` is per patient per fill. Annual packaging = packaging_cost_usd × 240 patients × fills_per_year.
- `reimbursement_per_fill_240_patients_usd` is already for all 240 patients per fill. Annual reimbursement = reimbursement_per_fill × fills_per_year.
- Drug cost: `(price_per_1000_doses_usd / 1000) × doses_per_fill × fills_per_year × patients`.
- Use `round(value, 2)` for all currency outputs.
- The JSON keys must match the schema EXACTLY (no extra keys, no missing keys).
- The `therapies` array must be sorted alphabetically by `therapy` name.
- The `recommendation.justification` should be a brief string explaining the decision referencing the threshold and the absolute difference.

### Step 3: Run the Script
```
python3 /root/solve.py
```

### Step 4: Validate Outputs
1. `cat /root/cycle_margin_analysis.json` — verify:
   - Valid JSON, parseable
   - All schema keys present
   - `assumptions` block matches given constants
   - `therapies` sorted alphabetically
   - All currency values rounded to 2 decimals
   - `totals` are correct sums
   - `recommendation.decision` is exactly `adopt_90_day` or `keep_30_day`

2. `cat /root/cycle_margin_summary.md` — verify:
   - 4-8 non-empty lines
   - Contains total 30-day margin USD value
   - Contains total 90-day margin USD value
   - Contains absolute difference USD value
   - Contains the exact slug `adopt_90_day` or `keep_30_day`

### Step 5: Cross-check one therapy manually
Pick one therapy and manually compute its values to verify the script is correct. Print intermediate values for that therapy to confirm drug cost, packaging cost, reimbursement, and margins match.

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