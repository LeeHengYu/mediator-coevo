# Task Instruction

## Task: Healthcare Cost-Benefit Analysis – 30-day vs 90-day Refill Cycle Margin Comparison

### Step 1: Inspect Input Files
Read and display the contents of all three input CSV files:
```
cat /root/acquisition_cost.csv
cat /root/packaging_cost.csv
cat /root/reimbursement.csv
```
Understand the column names, therapy names, and how they join (likely by therapy name and/or canister_size_units).

### Step 2: Write a Python Script to Perform the Analysis
Create `/root/solve.py` that does the following:

1. **Load CSVs** using the `csv` module (or `pandas` if available):
   - `acquisition_cost.csv` — contains at least `therapy`, `price_per_1000_doses_usd`
   - `packaging_cost.csv` — contains at least `therapy` (or similar), `canister_size_units`, `packaging_cost_usd`
   - `reimbursement.csv` — contains at least `therapy`, and a reimbursement column (reimbursement per fill for 240 patients)

2. **Merge/join** the three datasets by therapy name. Inspect column names carefully — they may vary. Use the actual column headers from the files.

3. **For each therapy, compute** (using the constants below):
   - `patients_per_therapy = 240`
   - `doses_per_fill_30 = 60`, `doses_per_fill_90 = 180`
   - `fills_per_year_30 = 12`, `fills_per_year_90 = 4`
   
   **Drug cost per fill** = `(doses_per_fill * price_per_1000_doses_usd) / 1000` — this is per-patient per fill.
   
   **Annual drug cost** = `drug_cost_per_fill * patients_per_therapy * fills_per_year`
   - `annual_drug_cost_30_day_usd = (60 / 1000) * price_per_1000_doses_usd * 240 * 12`
   - `annual_drug_cost_90_day_usd = (180 / 1000) * price_per_1000_doses_usd * 240 * 4`
   - NOTE: Both should yield the same total annual doses (240 patients × 2 inhalations/day × 365 days = 175,200 doses/year), so annual drug cost should be identical for 30-day and 90-day. Verify this.

   **Annual packaging cost**:
   - `packaging_cost_usd` is per patient per fill (from `packaging_cost.csv`, matched by `canister_size_units`)
   - `annual_packaging_cost_30_day_usd = packaging_cost_usd * 240 * 12`
   - `annual_packaging_cost_90_day_usd = packaging_cost_usd * 240 * 4`

   **Annual reimbursement**:
   - `reimbursement_per_fill_240_patients_usd` is the reimbursement per fill for all 240 patients combined (from `reimbursement.csv`)
   - `annual_reimbursement_30_day_usd = reimbursement_per_fill_240_patients_usd * 12`
   - `annual_reimbursement_90_day_usd = reimbursement_per_fill_240_patients_usd * 4`

   **Annual margin** = `annual_reimbursement - annual_drug_cost - annual_packaging_cost`
   
   **Margin difference** = `annual_margin_90_day_usd - annual_margin_30_day_usd`

4. **Sort therapies alphabetically** by `therapy` name.

5. **Compute totals**:
   - `total_annual_margin_30_day_usd` = sum of all therapies' `annual_margin_30_day_usd`
   - `total_annual_margin_90_day_usd` = sum of all therapies' `annual_margin_90_day_usd`
   - `total_annual_margin_difference_90_minus_30_usd` = sum of all per-therapy differences
   - `absolute_total_margin_difference_usd` = `abs(total_annual_margin_difference_90_minus_30_usd)`

6. **Decision rule**:
   - If `absolute_total_margin_difference_usd < 12000`, recommend `adopt_90_day`
   - Otherwise, recommend `keep_30_day`

7. **Round all currency values to 2 decimal places.**

8. **Write `/root/cycle_margin_analysis.json`** with the exact schema specified (use `json.dumps` with `indent=2`). Ensure:
   - `assumptions` block has exact keys and values as specified
   - `therapies` array is sorted alphabetically
   - `totals` block has all four keys
   - `recommendation` has `decision` (exact slug) and `justification` (a brief string)

9. **Write `/root/cycle_margin_summary.md`** with 4-8 non-empty lines containing:
   - Total 30-day margin (USD)
   - Total 90-day margin (USD)
   - Absolute difference (USD)
   - Final decision using exact slug (`adopt_90_day` or `keep_30_day`)

### Step 3: Run the Script
```
python3 /root/solve.py
```

### Step 4: Validate Outputs
```
cat /root/cycle_margin_analysis.json
cat /root/cycle_margin_summary.md
```

Verify:
- JSON is valid and parseable
- All currency values are rounded to 2 decimals
- Therapies array is sorted alphabetically by `therapy`
- The `assumptions` block matches exactly the specified values
- The summary has 4-8 non-empty lines and includes all required information with the exact decision slug
- The decision logic is correct: `abs(total_difference) < 12000` → `adopt_90_day`, otherwise `keep_30_day`

### Important Notes
- Inspect the CSV files carefully before coding. Column names and join keys may not be obvious.
- The packaging cost is matched by `canister_size_units` — this means you need to join acquisition_cost (which likely has canister_size_units) with packaging_cost on that field, and then join with reimbursement on therapy name.
- If any column names differ from expectations, adapt accordingly based on what you see in the actual files.
- Double-check that annual drug cost is the same for both models (since total annual doses = 240 × 730 = 175,200 regardless of fill cycle).
- Do NOT skip or hardcode any values. Compute everything from the CSV data.

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