# Task Instruction

## Task: Healthcare Cost-Benefit Analysis – 28-day vs 56-day Syncpack Comparison

### Objective
Read three CSV input files, compute annual margins for 28-day and 56-day medication synchronization card cycles, and produce two output files.

### Step 0 – Inspect Input Files
```bash
cat /root/ingredient_cost.csv
cat /root/card_cost.csv
cat /root/reimbursement.csv
```
Understand the column names, number of medications, and how they join (likely by medication name). Note the `blister_card_count` column in `card_cost.csv` – this is used to match rows to the correct card cost.

### Step 1 – Write a Python script `/root/solve.py`

Create a Python script that:

1. **Reads the three CSVs** using the `csv` module (or `pandas` if available).
2. **Joins/merges** the data by medication name. Each medication should appear in all three files.
3. **Computes per-medication values** using these constants:
   - `patients = 180`
   - `fills_28 = 12`, `fills_56 = 6`
   - `caps_per_fill_28 = 56`, `caps_per_fill_56 = 112`

   For each medication:
   - `annual_drug_cost_28 = (price_per_1000_capsules_usd / 1000) * caps_per_fill_28 * fills_28 * patients`
   - `annual_drug_cost_56 = (price_per_1000_capsules_usd / 1000) * caps_per_fill_56 * fills_56 * patients`
     *(Note: both should be identical since 56*12 == 112*6 == 672 capsules/patient/year. Compute them separately anyway.)*
   - `annual_packaging_cost_28 = card_cost_usd * patients * fills_28`
   - `annual_packaging_cost_56 = card_cost_usd * patients * fills_56`
     *Use the card_cost_usd that matches the medication's blister_card_count.*
   - `annual_reimbursement_28 = reimbursement_per_cycle_180_patients * fills_28`
   - `annual_reimbursement_56 = reimbursement_per_cycle_180_patients * fills_56`
   - `annual_margin_28 = annual_reimbursement_28 - annual_drug_cost_28 - annual_packaging_cost_28`
   - `annual_margin_56 = annual_reimbursement_56 - annual_drug_cost_56 - annual_packaging_cost_56`
   - `margin_diff = annual_margin_56 - annual_margin_28`

4. **Round all currency values to 2 decimal places.**

5. **Sort medications alphabetically** by medication name (case-insensitive sort, but preserve original casing).

6. **Compute totals:**
   - `total_annual_margin_28 = sum of all annual_margin_28`
   - `total_annual_margin_56 = sum of all annual_margin_56`
   - `total_diff = total_annual_margin_56 - total_annual_margin_28`
   - `abs_diff = abs(total_diff)`

7. **Decision rule:**
   - If `abs_diff < 9000` → `convert_to_56_day`
   - Otherwise → `keep_28_day`

8. **Write `/root/syncpack_analysis.json`** with the exact schema from the task. Use `json.dumps(..., indent=2)`. Include:
   - `assumptions` block with the fixed constants and `switch_threshold_usd: 9000`
   - `medications` array (sorted alphabetically)
   - `totals` block (all rounded to 2 decimals)
   - `recommendation` block with `decision` (exact slug) and a `justification` string that mentions the absolute difference and threshold.

9. **Write `/root/syncpack_summary.md`** with 4–8 non-empty lines containing:
   - Total 28-day margin (USD)
   - Total 56-day margin (USD)
   - Absolute difference (USD)
   - Final decision using the exact slug (`convert_to_56_day` or `keep_28_day`)

### Step 2 – Run the script
```bash
python3 /root/solve.py
```

### Step 3 – Validate outputs
```bash
cat /root/syncpack_analysis.json
```
Verify:
- JSON is valid and parseable.
- `medications` array is sorted alphabetically by `medication`.
- All currency fields have exactly 2 decimal places.
- `assumptions` block matches the fixed constants.
- `totals.total_annual_margin_difference_56_minus_28_usd` equals sum of per-med differences.
- `totals.absolute_total_margin_difference_usd` equals `abs(total_diff)`.
- `recommendation.decision` is one of the two exact slugs.

```bash
cat /root/syncpack_summary.md
```
Verify:
- 4–8 non-empty lines.
- Contains the total 28-day margin, 56-day margin, absolute difference, and exact decision slug.

### Important Notes
- The `card_cost.csv` may have multiple rows for different `blister_card_count` values. Match each medication to the correct card cost using `blister_card_count` from the medication's data (likely in `ingredient_cost.csv` or `card_cost.csv` itself). Inspect the files carefully in Step 0 to understand the join logic.
- Do NOT hardcode any medication-specific values. Read everything from the CSVs.
- Round at the per-medication level before summing totals, then round totals again to 2 decimals.
- The justification string should be informative (e.g., "The absolute margin difference of $X is below/above the $9,000 threshold, so we recommend ...").

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