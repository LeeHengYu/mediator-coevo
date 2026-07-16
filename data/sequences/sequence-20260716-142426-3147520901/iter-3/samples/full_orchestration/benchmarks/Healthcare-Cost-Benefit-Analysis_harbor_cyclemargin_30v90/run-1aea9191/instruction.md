# Task Instruction

## Task: Healthcare Cost-Benefit Analysis – 30-day vs 90-day Refill Cycle Margin Comparison

### Step 1: Inspect Input Files
Read and display the contents of all three input CSV files:
- `/root/acquisition_cost.csv`
- `/root/packaging_cost.csv`
- `/root/reimbursement.csv`

Note the column names, therapy names, and data types carefully.

### Step 2: Understand the Calculations

For each therapy and each model (30-day and 90-day):

**Constants:**
- patients_per_therapy = 240
- doses_per_day = 2
- 30-day model: doses_per_fill = 60, fills_per_year = 12
- 90-day model: doses_per_fill = 180, fills_per_year = 4

**Annual Drug Cost** (same for both models since total annual doses are the same):
- total_annual_doses_per_patient = doses_per_fill × fills_per_year (= 720 for both)
- annual_drug_cost = (price_per_1000_doses_usd / 1000) × doses_per_fill × fills_per_year × patients_per_therapy

Note: Since 60×12 = 180×4 = 720, annual drug cost is the same for 30-day and 90-day. Still compute both separately and round to 2 decimals.

**Annual Packaging Cost:**
- Match therapy to packaging_cost.csv by `canister_size_units`
- annual_packaging_cost = packaging_cost_usd × patients_per_therapy × fills_per_year
- 30-day: packaging_cost_usd × 240 × 12
- 90-day: packaging_cost_usd × 240 × 4

**Annual Reimbursement:**
- reimbursement_per_fill_240_patients_usd is from reimbursement.csv (this is per fill for all 240 patients)
- annual_reimbursement = reimbursement_per_fill_240_patients_usd × fills_per_year
- 30-day: reimbursement_per_fill × 12
- 90-day: reimbursement_per_fill × 4

**Annual Margin:**
- annual_margin = annual_reimbursement - annual_drug_cost - annual_packaging_cost

**Per-therapy difference:**
- annual_margin_difference_90_minus_30 = annual_margin_90_day - annual_margin_30_day

**Totals:**
- Sum all per-therapy margins for 30-day, 90-day, and differences
- absolute_total_margin_difference = abs(total_difference)

**Decision:**
- If abs(total_difference) < 12000 → "adopt_90_day"
- Otherwise → "keep_30_day"

### Step 3: Write a Python Script

Write and execute a Python script at `/root/solve.py` that:
1. Reads all three CSVs using pandas
2. Merges them by therapy name (inspect the actual join key — likely `therapy` column)
3. Also matches packaging by `canister_size_units`
4. Computes all values per the formulas above
5. Rounds all currency values to 2 decimal places
6. Sorts therapies alphabetically by therapy name
7. Outputs `/root/cycle_margin_analysis.json` with the EXACT schema specified (all keys must match exactly, all numeric values must be floats not strings)
8. Outputs `/root/cycle_margin_summary.md` with 4-8 non-empty lines containing: total 30-day margin, total 90-day margin, absolute difference, and the exact decision slug

### Step 4: Validate Outputs
1. Read and display `/root/cycle_margin_analysis.json` — verify:
   - `assumptions` has exactly the specified keys including `switch_threshold_usd: 12000`
   - `therapies` array is sorted alphabetically
   - All numeric fields are numbers (not strings)
   - All currency values have at most 2 decimal places
   - The schema matches exactly (no extra keys like 'currency' or 'units_cost_basis')
2. Read and display `/root/cycle_margin_summary.md` — verify:
   - 4-8 non-empty lines
   - Contains total 30-day margin, total 90-day margin, absolute difference, and the exact slug `adopt_90_day` or `keep_30_day`
3. Spot-check one therapy's calculations manually to confirm correctness

### Critical Warnings (from cross-task feedback):
- Do NOT add extra keys to the JSON schema (no 'currency', no 'units_cost_basis')
- Do NOT omit `switch_threshold_usd` from assumptions
- All numeric values in JSON must be actual numbers, NOT strings
- The summary must use numeric formatting that works with floats (don't store as strings)
- Double-check margin calculations carefully — errors in prior similar tasks caused verification failures

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