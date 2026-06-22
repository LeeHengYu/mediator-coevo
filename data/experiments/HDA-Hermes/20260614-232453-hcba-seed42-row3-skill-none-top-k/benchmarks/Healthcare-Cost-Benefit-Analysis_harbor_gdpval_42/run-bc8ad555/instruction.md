# Task Instruction

Execute the following steps in order:

1. **Inspect input files.** Read and display the contents of:
   - `/root/wholesale_price.csv`
   - `/root/vial_price.csv`
   - `/root/reimbursement.csv`
   Note the exact column names and medication names.

2. **Write a Python script** `/root/solve.py` that:
   a. Reads the three CSV files using pandas.
   b. Merges them on the medication name column (inspect the actual column names first; they may be `medication`, `drug`, etc.).
   c. For each of the 10 medications, computes:
      - `annual_drug_cost_90_day_usd` = (price_per_1000_tablets_usd / 1000) * 90 * 4 * 300
      - `annual_drug_cost_100_day_usd` = (price_per_1000_tablets_usd / 1000) * 100 * 3 * 300
      - `annual_supply_cost_90_day_usd` = vial_price_usd * 300 * 4
      - `annual_supply_cost_100_day_usd` = vial_price_usd * 300 * 3
      - `annual_reimbursement_90_day_usd` = reimbursement_per_fill_300_patients_usd * 4
      - `annual_reimbursement_100_day_usd` = reimbursement_per_fill_300_patients_usd * 3
      - `annual_revenue_90_day_usd` = annual_reimbursement_90_day_usd - annual_drug_cost_90_day_usd - annual_supply_cost_90_day_usd
      - `annual_revenue_100_day_usd` = annual_reimbursement_100_day_usd - annual_drug_cost_100_day_usd - annual_supply_cost_100_day_usd
      - `annual_revenue_difference_100_minus_90_usd` = annual_revenue_100_day_usd - annual_revenue_90_day_usd
   d. All currency values rounded to 2 decimal places.
   e. Computes totals:
      - `total_annual_revenue_90_day_usd` = sum of all annual_revenue_90_day_usd
      - `total_annual_revenue_100_day_usd` = sum of all annual_revenue_100_day_usd
      - `total_annual_revenue_difference_100_minus_90_usd` = sum of all per-medication differences
      - `absolute_total_revenue_difference_usd` = abs(total_annual_revenue_difference_100_minus_90_usd)
   f. Decision rule:
      - If `absolute_total_revenue_difference_usd < 16000`, decision = `"switch_to_100_day"`
      - Otherwise, decision = `"keep_90_day"`
   g. Builds the JSON object with **exactly** these top-level keys: `assumptions`, `medications`, `totals`, `recommendation`.
      - `assumptions` has exactly the six fields shown in the schema (patients_per_medication, fills_per_year_90_day, fills_per_year_100_day, tablets_per_fill_90_day, tablets_per_fill_100_day, switch_threshold_usd).
      - Each item in `medications` list has exactly these keys (with `_usd` suffixes): `medication`, `price_per_1000_tablets_usd`, `vial_size_drams`, `vial_price_usd`, `reimbursement_per_fill_300_patients_usd`, `annual_drug_cost_90_day_usd`, `annual_drug_cost_100_day_usd`, `annual_supply_cost_90_day_usd`, `annual_supply_cost_100_day_usd`, `annual_reimbursement_90_day_usd`, `annual_reimbursement_100_day_usd`, `annual_revenue_90_day_usd`, `annual_revenue_100_day_usd`, `annual_revenue_difference_100_minus_90_usd`.
      - `totals` has exactly: `total_annual_revenue_90_day_usd`, `total_annual_revenue_100_day_usd`, `total_annual_revenue_difference_100_minus_90_usd`, `absolute_total_revenue_difference_usd`.
      - `recommendation` has exactly: `decision` (the slug string) and `justification` (a brief explanation string).
   h. Writes the JSON to `/root/refill_analysis.json` with indent=2.
   i. Writes `/root/refill_summary.md` with 4-8 lines containing:
      - Total 90-day revenue in USD
      - Total 100-day revenue in USD
      - Absolute difference in USD
      - The exact decision slug (`switch_to_100_day` or `keep_90_day`)

3. **Run the script:** `python /root/solve.py`

4. **Validate the outputs:**
   a. Read `/root/refill_analysis.json` and verify:
      - Top-level keys are exactly `assumptions`, `medications`, `totals`, `recommendation`.
      - Every medication entry has all required `_usd`-suffixed keys.
      - `totals` has all four required keys including `absolute_total_revenue_difference_usd`.
      - `recommendation` has `decision` and `justification`.
      - All numeric values are rounded to 2 decimals.
   b. Read `/root/refill_summary.md` and verify it is 4-8 lines and includes the required information and the exact decision slug.
   c. If there is a test script at `/root/test_outputs.py`, run it: `python /root/test_outputs.py` and fix any failures.

5. **Critical reminders from prior failure:**
   - Every computed field name MUST end with `_usd`. Do NOT omit the suffix.
   - The `totals` keys must have the `total_` prefix (e.g., `total_annual_revenue_90_day_usd`, NOT `annual_revenue_90_day_usd`).
   - The top-level key must be `recommendation` (an object with `decision` and `justification`), NOT a bare `decision` string at the top level.
   - Match the schema character-for-character. Print the final JSON keys to stdout for visual confirmation before finishing.

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