# Task Instruction

## Task: Retail Pharmacy Auto-Refill Cost-Benefit Analysis

You must analyze whether a pharmacy should keep 90-day fills or switch to 100-day fills for its top 10 maintenance medications, then produce two output files.

### Step 1: Inspect Input Files

Read and display the contents of all three input CSV files:
```
cat /root/wholesale_price.csv
cat /root/vial_price.csv
cat /root/reimbursement.csv
```

Understand the column names and data types. There should be 10 medications across these files. Note the exact medication name strings — they must be preserved exactly in the output.

### Step 2: Write a Python Script

Create and run a Python script `/root/analyze.py` that:

1. **Reads** all three CSVs using `pandas` (or plain `csv` module).
2. **Joins** them on medication name. If column names differ slightly across files, inspect and handle accordingly.
3. **For each medication**, computes:
   - `annual_drug_cost_90_day = (price_per_1000_tablets / 1000) * 90 * 4 * 300`
   - `annual_drug_cost_100_day = (price_per_1000_tablets / 1000) * 100 * 3 * 300`
   - `annual_supply_cost_90_day = vial_price_usd * 300 * 4` (one vial per patient per fill)
   - `annual_supply_cost_100_day = vial_price_usd * 300 * 3`
   - `annual_reimbursement_90_day = reimbursement_per_fill_300_patients * 4`
   - `annual_reimbursement_100_day = reimbursement_per_fill_300_patients * 3`
   - `annual_revenue_90_day = annual_reimbursement_90_day - annual_drug_cost_90_day - annual_supply_cost_90_day`
   - `annual_revenue_100_day = annual_reimbursement_100_day - annual_drug_cost_100_day - annual_supply_cost_100_day`
   - `difference = annual_revenue_100_day - annual_revenue_90_day`
4. **All currency values** are rounded to 2 decimal places.
5. **Totals**: sum of all 10 medications' `annual_revenue_90_day`, `annual_revenue_100_day`, and `difference`. Also compute `abs(total_difference)`.
6. **Decision rule**: If `abs(total_difference) < 16000`, decision is `"switch_to_100_day"`. Otherwise `"keep_90_day"`.
7. **Justification**: A brief string explaining the decision referencing the absolute difference and the $16,000 threshold.

### Step 3: Write `/root/refill_analysis.json`

The JSON must follow the exact schema provided:
- `assumptions` block with the fixed values (patients_per_medication: 300, fills_per_year_90_day: 4, fills_per_year_100_day: 3, tablets_per_fill_90_day: 90, tablets_per_fill_100_day: 100, switch_threshold_usd: 16000)
- `medications` array with exactly 10 objects, each having ALL of these fields:
  - `medication` (string, exact name from CSV)
  - `price_per_1000_tablets_usd`
  - `vial_size_drams` (integer from vial_price.csv)
  - `vial_price_usd`
  - `reimbursement_per_fill_300_patients_usd`
  - `annual_drug_cost_90_day_usd`
  - `annual_drug_cost_100_day_usd`
  - `annual_supply_cost_90_day_usd`
  - `annual_supply_cost_100_day_usd`
  - `annual_reimbursement_90_day_usd`
  - `annual_reimbursement_100_day_usd`
  - `annual_revenue_90_day_usd`
  - `annual_revenue_100_day_usd`
  - `annual_revenue_difference_100_minus_90_usd`
- `totals` block with all four total fields
- `recommendation` block with `decision` (exactly `"switch_to_100_day"` or `"keep_90_day"`) and `justification`

Use `json.dump` with `indent=2` for readability.

### Step 4: Write `/root/refill_summary.md`

A markdown file, 4-8 lines, that includes:
- Total 90-day annual revenue in USD
- Total 100-day annual revenue in USD
- Absolute difference in USD
- The final decision using the exact slug: `switch_to_100_day` or `keep_90_day`

Example format:
```
# Refill Policy Analysis Summary

- Total 90-day annual revenue: $X.XX
- Total 100-day annual revenue: $Y.YY
- Absolute revenue difference: $Z.ZZ
- Recommendation: switch_to_100_day
```

### Step 5: Validate

1. Run: `python3 -c "import json; d=json.load(open('/root/refill_analysis.json')); print(len(d['medications']), 'medications'); print('totals:', d['totals']); print('decision:', d['recommendation']['decision'])"`
2. Run: `cat /root/refill_summary.md`
3. Verify there are exactly 10 medications in the JSON.
4. Verify the decision slug is one of the two exact allowed values.
5. Verify the summary is 4-8 lines and contains all required information.
6. Verify all numeric values are rounded to 2 decimal places.

### Important Notes
- The medication names in the JSON must match the CSV source exactly (case, spelling, spacing).
- The `vial_size_drams` should be an integer.
- Do NOT invent data. All values come from the three CSV files.
- The formulas above are exact — do not modify them.
- `switch_threshold_usd` in assumptions must be the integer `16000`, not a float.

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

Task-local resources are available under `environment/skills`: business-model-math-validation, loyalty-modeling, pharmacy-supply-chain, recursive-generosity-protocol, value-analysis.

# Verifier Contract

Success is judged by the SkillFlow verifier for this task.
Do not bypass, remove, or weaken verifier scripts, tests, fixtures, or expected-output checks.
Run the provided tests or verifier command when practical before finalizing.
Task metadata: author_email=codex@example.com, author_name=Codex, category=financial-analysis, difficulty=medium, tags=[pharmacy, unit-economics, cost-analysis, json, verification].
Verifier config: timeout_sec=900.0.