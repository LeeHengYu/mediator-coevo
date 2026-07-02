# Task Instruction

## Task: Healthcare Cost-Benefit Analysis — 28-day vs 56-day Syncpack

### Step 1: Inspect Input Files
Read and display the contents of all three input CSV files:
```
cat /root/ingredient_cost.csv
cat /root/card_cost.csv
cat /root/reimbursement.csv
```
Understand the columns, medication names, and how they join together.

### Step 2: Write and Run a Python Script
Create `/root/solve.py` that does the following:

1. **Load CSVs** using the `csv` module (or `pandas` if available).
2. **Join data** by medication name across the three files. Each medication should appear in all three files.
3. **For each medication**, compute:
   - `annual_drug_cost = (price_per_1000_capsules_usd / 1000) * capsules_per_fill * fills_per_year * 180`
     - 28-day: capsules_per_fill=56, fills_per_year=12
     - 56-day: capsules_per_fill=112, fills_per_year=6
     - NOTE: Both models yield the same annual drug cost (56*12 = 112*6 = 672 capsules/patient/year * 180 patients). Compute both anyway and round to 2 decimals.
   - `annual_packaging_cost = card_cost_usd * fills_per_year * 180`
     - Match `blister_card_count` from `card_cost.csv` to the medication.
     - 28-day: fills_per_year=12; 56-day: fills_per_year=6
   - `annual_reimbursement = reimbursement_per_cycle_180_patients_usd * fills_per_year`
     - 28-day: fills_per_year=12; 56-day: fills_per_year=6
   - `annual_margin = annual_reimbursement - annual_drug_cost - annual_packaging_cost` (for each model)
   - `annual_margin_difference_56_minus_28 = annual_margin_56_day - annual_margin_28_day`
   - Round ALL currency values to 2 decimal places.
4. **Sort medications alphabetically** by medication name (case-insensitive sort is fine, but match the original casing in output).
5. **Compute totals**:
   - `total_annual_margin_28_day_usd` = sum of all annual_margin_28_day_usd
   - `total_annual_margin_56_day_usd` = sum of all annual_margin_56_day_usd
   - `total_annual_margin_difference_56_minus_28_usd` = sum of all per-medication differences
   - `absolute_total_margin_difference_usd` = abs(total_annual_margin_difference_56_minus_28_usd)
   - Round all to 2 decimals.
6. **Decision rule**:
   - If `absolute_total_margin_difference_usd < 9000`, decision = `convert_to_56_day`
   - Otherwise, decision = `keep_28_day`
   - Write a short justification string that mentions the absolute difference and threshold.
7. **Write `/root/syncpack_analysis.json`** using `json.dumps` with `indent=2`. The schema must match EXACTLY what is specified in the task (all field names, nesting, types). Ensure numeric values are floats with 2 decimal places (use `round(x, 2)`).
8. **Write `/root/syncpack_summary.md`** with 4-8 non-empty lines containing:
   - Total 28-day margin (USD)
   - Total 56-day margin (USD)
   - Absolute difference (USD)
   - Final decision using the exact slug `convert_to_56_day` or `keep_28_day`

Run the script:
```
python3 /root/solve.py
```

### Step 3: Validate Outputs
```
cat /root/syncpack_analysis.json
cat /root/syncpack_summary.md
```

Verify:
- JSON is valid and parseable.
- `assumptions` block has the exact values specified.
- `medications` array is sorted alphabetically by `medication`.
- All currency fields are rounded to 2 decimal places.
- `totals` block sums match the individual medication entries.
- `recommendation.decision` is exactly one of the two slugs.
- The summary markdown has 4-8 non-empty lines and includes all four required pieces of information with the exact decision slug.
- Drug costs for 28-day and 56-day should be identical (both = price_per_1000 / 1000 * 672 * 180). Confirm this.
- Packaging costs for 56-day should be exactly half of 28-day packaging costs. Confirm this.

If anything is wrong, fix and re-run.

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