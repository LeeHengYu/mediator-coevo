# Task Instruction

This task was already solved successfully (reward 1.0) in the previous execution. Re-execute the same approach:

1. Read the three input CSV files:
   - `/root/acquisition_cost.csv` (contains `therapy`, `price_per_1000_doses_usd`)
   - `/root/packaging_cost.csv` (contains `canister_size_units`, `packaging_cost_usd`)
   - `/root/reimbursement.csv` (contains `therapy`, `reimbursement_per_fill_240_patients_usd`)

2. Inspect each CSV to understand column names and how they join (likely on `therapy` and/or `canister_size_units`).

3. Write a Python script that:
   a. Reads all three CSVs with pandas.
   b. Merges them on the appropriate keys (therapy name, canister_size_units for packaging).
   c. For each therapy, computes:
      - `annual_drug_cost = (price_per_1000_doses_usd / 1000) * doses_per_fill * fills_per_year * 240`
        - 30-day: doses_per_fill=60, fills_per_year=12
        - 90-day: doses_per_fill=180, fills_per_year=4
        - Note: annual drug cost should be the same for both (240 patients × 2 inhalations/day × 365 days worth), but compute per the formula structure.
      - `annual_packaging_cost = packaging_cost_usd * 240 * fills_per_year`
        - 30-day: fills_per_year=12
        - 90-day: fills_per_year=4
      - `annual_reimbursement = reimbursement_per_fill_240_patients_usd * fills_per_year`
        - 30-day: fills_per_year=12
        - 90-day: fills_per_year=4
      - `annual_margin = annual_reimbursement - annual_drug_cost - annual_packaging_cost`
      - `annual_margin_difference_90_minus_30 = margin_90 - margin_30`
   d. Sorts therapies alphabetically by `therapy`.
   e. Computes totals: sum of all 30-day margins, sum of all 90-day margins, total difference, absolute difference.
   f. Decision rule: if `abs(total_difference) < 12000` → `adopt_90_day`, else `keep_30_day`.
   g. Rounds all currency values to 2 decimal places.
   h. Writes `/root/cycle_margin_analysis.json` with the exact schema specified.
   i. Writes `/root/cycle_margin_summary.md` with 4-8 non-empty lines containing:
      - Total 30-day margin (USD) with comma-formatted numbers (e.g., `f'{value:,.2f}'`)
      - Total 90-day margin (USD) with comma-formatted numbers
      - Absolute difference (USD) with comma-formatted numbers
      - The exact decision slug (`adopt_90_day` or `keep_30_day`)

4. Run the script and verify:
   - `/root/cycle_margin_analysis.json` is valid JSON with the correct schema.
   - `/root/cycle_margin_summary.md` has 4-8 non-empty lines and contains the required information.
   - Currency values in the markdown use thousands separators (commas) per cross-task feedback.

IMPORTANT from cross-task feedback:
- Format currency in the markdown summary with comma separators: use `f'{value:,.2f}'` not just `f'{value:.2f}'`.
- Double-check that packaging cost is applied per patient per fill (240 patients × packaging_cost_usd × fills_per_year).
- Double-check reimbursement is already per fill for 240 patients (so just multiply by fills_per_year, NOT by 240 again).

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