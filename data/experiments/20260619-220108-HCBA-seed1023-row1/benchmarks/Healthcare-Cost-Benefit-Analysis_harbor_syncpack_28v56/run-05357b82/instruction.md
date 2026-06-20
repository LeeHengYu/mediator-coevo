# Task Instruction

## Task: Healthcare Cost-Benefit Analysis – 28-day vs 56-day Syncpack

### Step 1: Inspect Input Files
Read and display the contents of all three input CSV files:
```
cat /root/ingredient_cost.csv
cat /root/card_cost.csv
cat /root/reimbursement.csv
```
Understand the columns, medications listed, and how they join together.

### Step 2: Write a Python Script to Perform the Analysis
Create `/root/solve.py` that does the following:

1. **Load CSVs** using the `csv` module (or `pandas` if available).
2. **Join data** by medication name across the three files:
   - `ingredient_cost.csv` provides `medication` and `price_per_1000_capsules_usd`
   - `card_cost.csv` provides `medication`, `blister_card_count`, and `card_cost_usd`
   - `reimbursement.csv` provides `medication` and `reimbursement_per_cycle_180_patients_usd`
3. **For each medication, compute** (using the constants below):
   - `patients = 180`
   - `capsules_per_fill_28 = 56`, `fills_28 = 12`
   - `capsules_per_fill_56 = 112`, `fills_56 = 6`
   - `annual_drug_cost_28 = (price_per_1000_capsules_usd / 1000) * capsules_per_fill_28 * fills_28 * patients`
   - `annual_drug_cost_56 = (price_per_1000_capsules_usd / 1000) * capsules_per_fill_56 * fills_56 * patients`
   - `annual_packaging_cost_28 = card_cost_usd * fills_28 * patients`
   - `annual_packaging_cost_56 = card_cost_usd * fills_56 * patients`
   - `annual_reimbursement_28 = reimbursement_per_cycle_180_patients_usd * fills_28`
   - `annual_reimbursement_56 = reimbursement_per_cycle_180_patients_usd * fills_56`
   - `annual_margin_28 = annual_reimbursement_28 - annual_drug_cost_28 - annual_packaging_cost_28`
   - `annual_margin_56 = annual_reimbursement_56 - annual_drug_cost_56 - annual_packaging_cost_56`
   - `margin_difference = annual_margin_56 - annual_margin_28`
4. **Round all currency values to 2 decimal places** using `round(value, 2)`.
5. **Sort medications alphabetically** by `medication` (case-sensitive standard sort).
6. **Compute totals**:
   - `total_annual_margin_28 = sum of all annual_margin_28`
   - `total_annual_margin_56 = sum of all annual_margin_56`
   - `total_difference = sum of all margin_difference` (equivalently total_56 - total_28)
   - `absolute_total = abs(total_difference)`
   - Round all totals to 2 decimals.
7. **Decision rule**:
   - If `abs(total_difference) < 9000` → `"convert_to_56_day"`
   - Otherwise → `"keep_28_day"`
8. **Write `/root/syncpack_analysis.json`** with the exact schema specified. Use `json.dump` with `indent=2`. The `justification` string should be a short sentence referencing the absolute difference and the $9,000 threshold.
9. **Write `/root/syncpack_summary.md`** with 4–8 non-empty lines including:
   - Total 28-day margin (USD)
   - Total 56-day margin (USD)
   - Absolute difference (USD)
   - Final decision using the exact slug `convert_to_56_day` or `keep_28_day`

### Step 3: Run the Script
```
python3 /root/solve.py
```

### Step 4: Validate Outputs
1. Read and display `/root/syncpack_analysis.json`. Verify:
   - `assumptions` block has the exact keys and values specified.
   - `medications` array is sorted alphabetically by `medication`.
   - All currency fields are rounded to 2 decimal places.
   - Each medication has all 14 required fields.
   - `totals` block has 4 fields, all rounded to 2 decimals.
   - `recommendation.decision` is one of the two exact slugs.
   - The JSON is valid (parseable).
2. Read and display `/root/syncpack_summary.md`. Verify:
   - 4–8 non-empty lines.
   - Contains total 28-day margin, total 56-day margin, absolute difference, and the decision slug.
3. Spot-check one medication's math manually:
   - Pick the first medication alphabetically.
   - Recompute annual_drug_cost_28 by hand from the CSV values.
   - Confirm it matches the JSON output.

### Important Notes
- Drug cost formula: note that both 28-day and 56-day models use the same total capsules per year (56×12 = 672 vs 112×6 = 672 per patient), so annual drug costs should be identical for both models. This is expected and correct per the spec.
- Packaging costs differ because the number of fills differs (12 vs 6), so 56-day model has half the packaging cost.
- Reimbursement differs because it's per-cycle and the number of cycles differs (12 vs 6).
- Do NOT modify the decision threshold or rounding rules.
- The `blister_card_count` field from `card_cost.csv` should be included in the output JSON for each medication but is used only for matching/informational purposes; the `card_cost_usd` is the per-patient-per-fill packaging cost.

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