# Task Instruction

You are completing a healthcare cost-benefit analysis task. The previous execution scored 1.0, so replicate the same approach.

Step 1: Read all input files:
- `cat /root/assay_manifest.json`
- `cat /root/carrier_cost.csv`
- `cat /root/billing.csv`
- `cat /root/lab_overrides.csv`
- `cat /root/report_template.json`

Step 2: Write and run a Python script `/root/solve.py` that does the following:

1. **Load data**: Read `assay_manifest.json`, `carrier_cost.csv`, `billing.csv`, `lab_overrides.csv`, and `report_template.json`.

2. **Filter in-scope assays**: From `assay_manifest.json`, keep only assays where `in_scope` is `true`.

3. **Resolve billing rows**:
   - For each billing row, match `assay_label` to either `assay_name` or any alias in the assay's `aliases` list from the manifest. This maps each billing row to an `assay_id`.
   - Keep only rows where `is_active` is `true` (handle both boolean and string representations).
   - If multiple active billing rows map to the same assay, keep the one with the latest `effective_month` (compare as strings, YYYY-MM format).

4. **Resolve active labs**:
   - From `lab_overrides.csv`, keep only rows where `status` is `approved`.
   - If multiple approved rows exist for the same `assay_id`, keep the one with the highest `revision` number.
   - If an in-scope assay has no approved override, use `default_active_labs` from its manifest entry.

5. **Compute per-assay financials** for each in-scope assay:
   - `carrier_cost_usd` from `carrier_cost.csv` matched by `carrier_type` from the manifest.
   - Small-kit model: `runs_per_year = 24`, `tests_per_lab_per_run = tests_per_lab_per_run_small`
   - Bulk-kit model: `runs_per_year = 12`, `tests_per_lab_per_run = tests_per_lab_per_run_bulk`
   - `annual_revenue = payment_per_run_per_lab_usd * active_labs * runs_per_year`
   - `annual_reagent_cost = reagent_price_per_1000_tests_usd * active_labs * tests_per_lab_per_run * runs_per_year / 1000`
   - `annual_carrier_cost = carrier_cost_usd * active_labs * runs_per_year`
   - `annual_margin = annual_revenue - annual_reagent_cost - annual_carrier_cost`
   - `annual_margin_difference_bulk_minus_small = annual_margin_bulk - annual_margin_small`
   - Round ALL currency values to 2 decimal places.

6. **Compute totals**:
   - Sum all per-assay margins for small-kit and bulk-kit.
   - `total_difference = total_bulk_margin - total_small_margin`
   - `absolute_total_margin_difference_usd = abs(total_difference)`
   - Round to 2 decimals.

7. **Decision rule**:
   - If `abs(total_difference) < 7000`, recommend `adopt_bulk_kit`.
   - Otherwise, recommend `keep_small_kit`.
   - Write a short justification string.

8. **Output `/root/reagent_policy_report.json`**:
   - Preserve `metadata` from `report_template.json` exactly as-is.
   - Use the schema from the task. Sort `analysis.assays` by `assay_id` ascending.
   - Include `assumptions` with exact keys: `runs_per_year_small_kit: 24`, `runs_per_year_bulk_kit: 12`, `switch_threshold_usd: 7000`, `lab_override_rule: "highest approved revision per assay_id, else default_active_labs"`, `billing_rule: "latest active effective_month per assay"`.

9. **Output `/root/reagent_policy_summary.md`**:
   - 4-8 non-empty lines.
   - Include total small-kit margin (USD with commas), total bulk-kit margin (USD with commas), absolute difference (USD with commas), and the exact decision slug (`adopt_bulk_kit` or `keep_small_kit`).

Step 3: Run `python3 /root/solve.py` and verify it completes without errors.

Step 4: Verify outputs:
- `cat /root/reagent_policy_report.json` — check JSON is valid, assays sorted by assay_id, metadata preserved, all fields present with 2-decimal currency values.
- `cat /root/reagent_policy_summary.md` — check 4-8 non-empty lines with required content.

Step 5: If a test script exists at `/root/test_output.py` or similar, run `cd /root && python -m pytest test_output.py -v` to confirm passing.

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
Task metadata: author_email=gpt54@example.com, author_name=GPT-5.4, category=financial-analysis, difficulty=medium, tags=[lab-operations, json, csv, template-update, decision-analysis].
Verifier config: timeout_sec=900.0.