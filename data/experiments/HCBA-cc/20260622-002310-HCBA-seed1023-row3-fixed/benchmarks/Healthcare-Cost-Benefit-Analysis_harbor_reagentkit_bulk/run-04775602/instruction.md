# Task Instruction

Execute the following steps in order:

1. **Inspect all input files** to understand their structure and content:
 ```
 cat /root/assay_manifest.json
 cat /root/carrier_cost.csv
 cat /root/billing.csv
 cat /root/lab_overrides.csv
 cat /root/report_template.json
 ```

2. **Inspect the verifier** to understand what tests will be run:
 ```
 cat /root/test_output.py
 ```
 If it doesn't exist, check for any `.py` test files in `/root/`.

3. **Write a Python script** `/root/solve.py` that performs the full analysis. The script must:

 a. Load all input files:
 - `assay_manifest.json` — parse as JSON
 - `carrier_cost.csv` — parse as CSV
 - `billing.csv` — parse as CSV
 - `lab_overrides.csv` — parse as CSV
 - `report_template.json` — parse as JSON (preserve `metadata` exactly)

 b. Filter to in-scope assays: only those with `in_scope` == `true` in the manifest.

 c. For each in-scope assay, resolve the billing row:
 - Match `billing.csv` rows where `assay_label` equals the assay's `assay_name` OR any alias in the assay's alias list.
 - Keep only rows where `is_active` is `true` (handle string/bool variants: "true", "True", True, etc.).
 - If multiple active rows match the same assay, keep the one with the latest `effective_month` (lexicographic string comparison works for YYYY-MM format).
 - Extract `payment_per_run_per_lab_usd` from the retained row.

 d. Resolve active lab count per assay:
 - From `lab_overrides.csv`, keep only rows where `status` is `approved` (case-sensitive match as found in file).
 - If multiple approved rows exist for the same `assay_id`, keep the one with the highest `revision` number.
 - If no approved override row exists for an in-scope assay, use `default_active_labs` from the manifest.
 - The override row should contain an `active_labs` (or similarly named) field — inspect the CSV header to find the correct column name.

 e. Look up carrier cost:
 - Each assay has a `carrier_type` in the manifest.
 - Match to `carrier_cost.csv` by `carrier_type` to get `carrier_cost_usd`.
 - **IMPORTANT**: `carrier_cost_usd` is a per-shipment cost. Annual carrier cost = `carrier_cost_usd * active_labs * runs_per_year`. Inspect the verifier or test to confirm whether carrier cost scales by labs and runs, or just runs, or is flat. If the test file gives hints, follow that. If ambiguous, use: `annual_carrier_cost = carrier_cost_usd * active_labs * runs_per_year`.

 f. Compute per-assay financials (all rounded to 2 decimals at the end):
 - `annual_revenue_small = payment_per_run_per_lab_usd * active_labs * 24`
 - `annual_revenue_bulk = payment_per_run_per_lab_usd * active_labs * 12`
 - `annual_reagent_cost_small = reagent_price_per_1000_tests_usd * active_labs * tests_per_lab_per_run_small * 24 / 1000`
 - `annual_reagent_cost_bulk = reagent_price_per_1000_tests_usd * active_labs * tests_per_lab_per_run_bulk * 12 / 1000`
 - `annual_carrier_cost_small = carrier_cost_usd * active_labs * 24`
 - `annual_carrier_cost_bulk = carrier_cost_usd * active_labs * 12`
 - `annual_margin_small = annual_revenue_small - annual_reagent_cost_small - annual_carrier_cost_small`
 - `annual_margin_bulk = annual_revenue_bulk - annual_reagent_cost_bulk - annual_carrier_cost_bulk`
 - `difference = annual_margin_bulk - annual_margin_small`
 - Round each of these to 2 decimal places.

 g. Sort assays by `assay_id` ascending.

 h. Compute totals:
 - `total_annual_margin_small_kit_usd` = sum of all per-assay `annual_margin_small_kit_usd`
 - `total_annual_margin_bulk_kit_usd` = sum of all per-assay `annual_margin_bulk_kit_usd`
 - `total_annual_margin_difference_bulk_minus_small_usd` = sum of all per-assay differences
 - `absolute_total_margin_difference_usd` = abs(total_difference)
 - Round all to 2 decimals.

 i. Decision:
 - If `abs(total_difference) < 7000`, decision = `adopt_bulk_kit`
 - Otherwise, decision = `keep_small_kit`
 - Write a brief justification string.

 j. Build the output JSON matching the schema exactly. Include `metadata` from `report_template.json` as-is.

 k. Write `/root/reagent_policy_report.json` with `json.dump(..., indent=2)`.

 l. Write `/root/reagent_policy_summary.md` with 4-8 non-empty lines including:
 - Total small-kit margin (USD)
 - Total bulk-kit margin (USD)
 - Absolute difference (USD)
 - Final decision using exact slug (`adopt_bulk_kit` or `keep_small_kit`)

 **CRITICAL NOTES based on past failures:**
 - Do NOT confuse carrier cost scaling. Read the task formula carefully: the task says `annual_carrier_cost` but does not give an explicit formula for it. The most natural interpretation given the other formulas is `carrier_cost_usd * active_labs * runs_per_year`. However, if the test expects `carrier_cost_usd * runs_per_year` (without labs), adjust accordingly. **First try with `carrier_cost_usd * active_labs * runs_per_year`**. If the test fails, inspect the expected values and adjust.
 - Parse boolean fields carefully from CSVs (they may be strings like "true"/"false" or "True"/"False").
 - Parse numeric fields from CSVs carefully (they may have quotes or whitespace).
 - The `effective_month` comparison should be string-based (YYYY-MM format sorts correctly lexicographically).

4. **Run the script:**
 ```
 cd /root && python solve.py
 ```

5. **Verify the outputs exist and look correct:**
 ```
 cat /root/reagent_policy_report.json
 cat /root/reagent_policy_summary.md
 ```

6. **Run the verifier test:**
 ```
 cd /root && python -m pytest test_output.py -v
 ```
 If no test file exists, skip this step.

7. **If tests fail**, read the error messages carefully. Common issues:
 - Carrier cost formula (with/without labs multiplier)
 - Wrong column name from lab_overrides.csv for active labs count
 - Boolean parsing issues
 - Alias matching issues
 - Rounding issues
 Fix the script and re-run until tests pass.

8. **After tests pass**, confirm both output files are in place and the task is complete.

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