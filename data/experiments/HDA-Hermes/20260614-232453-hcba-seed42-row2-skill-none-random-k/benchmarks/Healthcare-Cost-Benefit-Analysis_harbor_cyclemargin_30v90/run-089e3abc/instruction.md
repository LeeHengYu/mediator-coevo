# Task Instruction

Execute the following steps in order:

1. **Inspect input files.** Read and display the contents of:
   - `/root/acquisition_cost.csv`
   - `/root/packaging_cost.csv`
   - `/root/reimbursement.csv`

2. **Write and run a Python script** (`/root/solve.py`) that does the following:

   a. Parse all three CSVs. Join them by therapy name. The packaging CSV has a `canister_size_units` column; match it to the acquisition cost row for each therapy.

   b. Constants:
      - `patients = 240`
      - `doses_per_fill_30 = 60`, `fills_per_year_30 = 12`
      - `doses_per_fill_90 = 180`, `fills_per_year_90 = 4`
      - `switch_threshold = 12000`

   c. For each therapy compute:
      - `annual_drug_cost_30 = (price_per_1000_doses / 1000) * doses_per_fill_30 * fills_per_year_30 * patients`
      - `annual_drug_cost_90 = (price_per_1000_doses / 1000) * doses_per_fill_90 * fills_per_year_90 * patients`
      - `annual_packaging_cost_30 = packaging_cost_usd * patients * fills_per_year_30`
      - `annual_packaging_cost_90 = packaging_cost_usd * patients * fills_per_year_90`
      - `annual_reimbursement_30 = reimbursement_per_fill_240_patients * fills_per_year_30`
      - `annual_reimbursement_90 = reimbursement_per_fill_240_patients * fills_per_year_90`
      - `annual_margin_30 = annual_reimbursement_30 - annual_drug_cost_30 - annual_packaging_cost_30`
      - `annual_margin_90 = annual_reimbursement_90 - annual_drug_cost_90 - annual_packaging_cost_90`
      - `margin_diff = annual_margin_90 - annual_margin_30`
      - Round all currency values to 2 decimal places.

   d. Sort therapies alphabetically by therapy name.

   e. Compute totals:
      - `total_annual_margin_30 = sum of all annual_margin_30`
      - `total_annual_margin_90 = sum of all annual_margin_90`
      - `total_diff = total_annual_margin_90 - total_annual_margin_30`
      - `abs_diff = abs(total_diff)`
      - Round all to 2 decimals.

   f. Decision: if `abs_diff < 12000` then `adopt_90_day`, else `keep_30_day`.

   g. Build the JSON object exactly matching the schema in the task (field names must match exactly). Write to `/root/cycle_margin_analysis.json` with `json.dump(..., indent=2)`.

   h. Write `/root/cycle_margin_summary.md` with 4-8 non-empty lines that include:
      - Total 30-day margin with USD amount
      - Total 90-day margin with USD amount
      - Absolute difference with USD amount
      - The exact decision slug (`adopt_90_day` or `keep_30_day`)

3. **Run the script:** `python3 /root/solve.py`

4. **Validate outputs:**
   - `cat /root/cycle_margin_analysis.json` and verify it parses, has all required keys, therapies are alphabetically sorted, and all values are rounded to 2 decimals.
   - `cat /root/cycle_margin_summary.md` and verify it has 4-8 non-empty lines and contains the required information.
   - If a test file exists (e.g., `test_output.py`), run `cd /root && python3 -m pytest test_output.py -v` and confirm all tests pass.

5. **If any test fails**, read the error message carefully, fix the script or outputs, and re-run until all tests pass.

Key cautions from cross-task feedback:
- Ensure the summary markdown uses the exact decision slug (`adopt_90_day` or `keep_30_day`), not a paraphrase.
- Double-check that 90-day cost calculations use `doses_per_fill_90 * fills_per_year_90` (= 720 doses/year, same as 30-day: 60*12=720), so annual drug costs should be identical between 30 and 90-day models. The margin difference is driven by packaging cost reduction and reimbursement changes.
- Match packaging costs by `canister_size_units` if needed — inspect the CSVs first to understand the join keys.

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