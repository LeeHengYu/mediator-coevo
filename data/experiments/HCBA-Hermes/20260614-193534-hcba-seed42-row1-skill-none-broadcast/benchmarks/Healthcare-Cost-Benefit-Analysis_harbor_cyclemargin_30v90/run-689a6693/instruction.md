# Task Instruction

Execute the following steps in order:

1. **Inspect the input files** to understand their structure and contents:
 ```
 cat /root/acquisition_cost.csv
 cat /root/packaging_cost.csv
 cat /root/reimbursement.csv
 ```

2. **Write and run a Python script** (`/root/solve.py`) that does everything below:

 a. **Read the three CSVs** using the `csv` module (or pandas). The key columns are:
 - `acquisition_cost.csv`: `therapy`, `price_per_1000_doses_usd`
 - `packaging_cost.csv`: `therapy`, `canister_size_units`, `packaging_cost_usd`
 - `reimbursement.csv`: `therapy`, `reimbursement_per_fill_240_patients_usd`

 b. **Merge/join** the three datasets on `therapy`.

 c. **For each therapy, compute** (using the constants: patients=240, 30-day: 60 doses/fill, 12 fills/year; 90-day: 180 doses/fill, 4 fills/year):

 - `annual_drug_cost_30_day_usd` = `price_per_1000_doses_usd / 1000 * 60 * 240 * 12`
 - `annual_drug_cost_90_day_usd` = `price_per_1000_doses_usd / 1000 * 180 * 240 * 4`
 - `annual_packaging_cost_30_day_usd` = `packaging_cost_usd * 240 * 12`
 - `annual_packaging_cost_90_day_usd` = `packaging_cost_usd * 240 * 4`
 - `annual_reimbursement_30_day_usd` = `reimbursement_per_fill_240_patients_usd * 12`
 - `annual_reimbursement_90_day_usd` = `reimbursement_per_fill_240_patients_usd * 4`
 - `annual_margin_30_day_usd` = `annual_reimbursement_30_day - annual_drug_cost_30_day - annual_packaging_cost_30_day`
 - `annual_margin_90_day_usd` = `annual_reimbursement_90_day - annual_drug_cost_90_day - annual_packaging_cost_90_day`
 - `annual_margin_difference_90_minus_30_usd` = `annual_margin_90_day_usd - annual_margin_30_day_usd`

 d. **Round all currency values to 2 decimal places.**

 e. **Sort therapies alphabetically** by `therapy` name (case-sensitive standard sort).

 f. **Compute totals:**
 - `total_annual_margin_30_day_usd` = sum of all `annual_margin_30_day_usd`
 - `total_annual_margin_90_day_usd` = sum of all `annual_margin_90_day_usd`
 - `total_annual_margin_difference_90_minus_30_usd` = sum of all `annual_margin_difference_90_minus_30_usd`
 - `absolute_total_margin_difference_usd` = `abs(total_annual_margin_difference_90_minus_30_usd)`

 g. **Decision rule:**
 - If `absolute_total_margin_difference_usd < 12000`, decision = `"adopt_90_day"`
 - Otherwise, decision = `"keep_30_day"`
 - Write a short justification string that mentions the absolute difference and threshold.

 h. **Write `/root/cycle_margin_analysis.json`** with the exact schema from the task (use `json.dumps` with `indent=2`). Make sure:
 - The `assumptions` block has the exact keys and values specified.
 - Each therapy object has all 14 fields including `therapy`, `price_per_1000_doses_usd`, `canister_size_units`, `packaging_cost_usd`, `reimbursement_per_fill_240_patients_usd`, and all computed fields.
 - `canister_size_units` should be an integer.
 - All USD values are floats rounded to 2 decimals.

 i. **Write `/root/cycle_margin_summary.md`** with 4-8 non-empty lines containing:
 - Total 30-day margin (USD) with the numeric value
 - Total 90-day margin (USD) with the numeric value
 - Absolute difference (USD) with the numeric value
 - Final decision using the exact slug `adopt_90_day` or `keep_30_day`

3. **Run the script:**
 ```
 python3 /root/solve.py
 ```

4. **Validate the outputs:**
 ```
 cat /root/cycle_margin_analysis.json
 cat /root/cycle_margin_summary.md
 ```
 Confirm:
 - JSON is valid and parseable.
 - `therapies` array is sorted alphabetically by `therapy`.
 - All currency values have 2 decimal places.
 - The `totals` and `recommendation` sections are present and consistent.
 - The summary markdown has 4-8 non-empty lines and includes all four required pieces of information with the exact decision slug.
 - The `absolute_total_margin_difference_usd` in totals matches `abs(total_annual_margin_difference_90_minus_30_usd)`.
 - The decision is consistent with the threshold rule.

5. **If any validation fails**, fix the script and re-run until both output files are correct.

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