# Task Instruction

Execute the following steps exactly, in order.

## Step 0 – Inspect all input files and the test suite

```bash
cat /root/assay_manifest.json
cat /root/carrier_cost.csv
cat /root/billing.csv
cat /root/lab_overrides.csv
cat /root/report_template.json
# Also inspect the test file so we know exactly what the verifier checks
find /root /tests -name '*.py' -path '*/test*' 2>/dev/null | head -20
for f in $(find /root /tests -name '*.py' -path '*/test*' 2>/dev/null); do cat "$f"; done
```

Read every file carefully before writing any code.

## Step 1 – Write and run a single Python script `/root/solve.py`

Create `/root/solve.py` that:

1. Loads `assay_manifest.json`, `carrier_cost.csv`, `billing.csv`, `lab_overrides.csv`, `report_template.json`.
2. Filters in-scope assays (`in_scope == true`).
3. For each in-scope assay:
   a. Resolve billing rows: match `assay_label` in billing.csv to `assay_name` OR any alias in the manifest for that assay. Keep only rows with `is_active` == true (handle string/bool). If multiple active rows map to the same assay, keep the one with the latest `effective_month`.
   b. Resolve active labs: from `lab_overrides.csv`, keep rows where `status` == `approved`. If multiple approved rows for the same `assay_id`, keep the one with the highest `revision`. If no approved row exists for an in-scope assay, use `default_active_labs` from the manifest.
   c. Look up `carrier_cost_usd` from `carrier_cost.csv` by matching the assay's `carrier_type`.
   d. Compute for small-kit (runs_per_year=24, tests_per_lab_per_run from `tests_per_lab_per_run_small`):
      - `annual_revenue_small = payment_per_run_per_lab_usd * active_labs * 24`
      - `annual_reagent_cost_small = reagent_price_per_1000_tests_usd * active_labs * tests_per_lab_per_run_small * 24 / 1000`
      - `annual_carrier_cost_small = carrier_cost_usd * active_labs * 24`
      - `annual_margin_small = annual_revenue_small - annual_reagent_cost_small - annual_carrier_cost_small`
   e. Compute for bulk-kit (runs_per_year=12, tests_per_lab_per_run from `tests_per_lab_per_run_bulk`):
      - `annual_revenue_bulk = payment_per_run_per_lab_usd * active_labs * 12`
      - `annual_reagent_cost_bulk = reagent_price_per_1000_tests_usd * active_labs * tests_per_lab_per_run_bulk * 12 / 1000`
      - `annual_carrier_cost_bulk = carrier_cost_usd * active_labs * 12`
      - `annual_margin_bulk = annual_revenue_bulk - annual_reagent_cost_bulk - annual_carrier_cost_bulk`
   f. `difference = annual_margin_bulk - annual_margin_small`
   g. Round ALL currency values to 2 decimal places.
4. Sort assays by `assay_id` ascending.
5. Compute totals:
   - `total_annual_margin_small_kit_usd` = sum of all per-assay `annual_margin_small_kit_usd`, rounded to 2 decimals.
   - `total_annual_margin_bulk_kit_usd` = sum of all per-assay `annual_margin_bulk_kit_usd`, rounded to 2 decimals.
   - `total_annual_margin_difference_bulk_minus_small_usd` = sum of all per-assay differences, rounded to 2 decimals.
   - `absolute_total_margin_difference_usd` = abs(total_annual_margin_difference_bulk_minus_small_usd), rounded to 2 decimals.
6. Decision: if `absolute_total_margin_difference_usd < 7000`, decision = `adopt_bulk_kit`; else `keep_small_kit`.
7. Build the output JSON with EXACTLY these keys in the `totals` dict:
   - `total_annual_margin_small_kit_usd`
   - `total_annual_margin_bulk_kit_usd`
   - `total_annual_margin_difference_bulk_minus_small_usd`
   - `absolute_total_margin_difference_usd`
8. The `metadata` object must be copied verbatim from `report_template.json`.
9. The `assumptions` object must contain exactly:
   ```json
   {
     "runs_per_year_small_kit": 24,
     "runs_per_year_bulk_kit": 12,
     "switch_threshold_usd": 7000,
     "lab_override_rule": "highest approved revision per assay_id, else default_active_labs",
     "billing_rule": "latest active effective_month per assay"
   }
   ```
10. Each assay object must contain exactly these keys (use these exact names):
    `assay_id`, `assay_name`, `active_labs`, `reagent_price_per_1000_tests_usd`, `carrier_type`, `carrier_cost_usd`, `payment_per_run_per_lab_usd`, `tests_per_lab_per_run_small`, `tests_per_lab_per_run_bulk`, `annual_reagent_cost_small_kit_usd`, `annual_reagent_cost_bulk_kit_usd`, `annual_carrier_cost_small_kit_usd`, `annual_carrier_cost_bulk_kit_usd`, `annual_revenue_small_kit_usd`, `annual_revenue_bulk_kit_usd`, `annual_margin_small_kit_usd`, `annual_margin_bulk_kit_usd`, `annual_margin_difference_bulk_minus_small_usd`.
11. Write `/root/reagent_policy_report.json` with `json.dump(..., indent=2)`.
12. Write `/root/reagent_policy_summary.md` with 4–8 non-empty lines including:
    - Total small-kit margin (USD)
    - Total bulk-kit margin (USD)
    - Absolute difference (USD)
    - Final decision using the exact slug `adopt_bulk_kit` or `keep_small_kit`

**IMPORTANT implementation notes:**
- When matching `is_active` in billing.csv, handle both boolean True and string "true"/"True" (CSV readers often return strings).
- When matching `assay_label` to aliases, check if the manifest stores aliases as a list. Compare case-insensitively if needed, but first check exact match.
- The `carrier_cost.csv` `carrier_cost_usd` field: parse as float.
- The annual carrier cost formula is: `carrier_cost_usd * active_labs * runs_per_year`. (Each lab incurs carrier cost per run.)
- Print intermediate values for debugging.

## Step 2 – Run the script

```bash
cd /root && python solve.py
```

Fix any errors and re-run until it completes.

## Step 3 – Validate outputs

```bash
cat /root/reagent_policy_report.json
cat /root/reagent_policy_summary.md
```

Verify:
- JSON is valid and parseable.
- `totals` has all four required keys.
- `assays` are sorted by `assay_id`.
- `metadata` matches `report_template.json` exactly.
- `assumptions` has the exact values specified.
- Summary has 4–8 non-empty lines with required content.

## Step 4 – Run the test suite

```bash
cd / && python -m pytest /tests/test_output*.py -v 2>&1 || python -m pytest /tests/ -v 2>&1
```

If any tests fail, read the error messages carefully, fix the issue in `solve.py`, re-run, and re-test. Do NOT modify test files.

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