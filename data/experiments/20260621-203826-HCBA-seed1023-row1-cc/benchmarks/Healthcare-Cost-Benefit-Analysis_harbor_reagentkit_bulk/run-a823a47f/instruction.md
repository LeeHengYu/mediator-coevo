# Task Instruction

Execute the following steps in order:

## 1. Inspect all input files and the test suite

```bash
cat /root/assay_manifest.json
cat /root/carrier_cost.csv
cat /root/billing.csv
cat /root/lab_overrides.csv
cat /root/report_template.json
cat /tests/test_outputs.py
```

Read every file carefully. Pay special attention to:
- The test file (`test_outputs.py`) — it defines the exact schema, keys, values, and string literals the verifier expects. The previous run failed because `assumptions` was missing keys. The test expects **exactly 8 keys** in `assumptions`, including keys like `annual_margin_formula`, `carrier_cost_interpretation`, and `decision_rule` with specific string values. Extract every expected key and value from the test assertions.
- The `report_template.json` — its `metadata` object must be preserved exactly as-is.
- The `assay_manifest.json` — note `in_scope`, `aliases`, `default_active_labs`, `tests_per_lab_per_run_small`, `tests_per_lab_per_run_bulk`, `reagent_price_per_1000_tests_usd`, `carrier_type`.

## 2. Build the computation in a Python script

Create `/root/solve.py` that:

### 2a. Load data
- Load `assay_manifest.json`, `carrier_cost.csv`, `billing.csv`, `lab_overrides.csv`, `report_template.json`.

### 2b. Filter in-scope assays
- Keep only assays where `in_scope` is `true`.

### 2c. Resolve billing rows
- For each in-scope assay, find billing rows where `assay_label` matches either `assay_name` or any alias in the manifest entry.
- Keep only rows where `is_active` is `true` (handle string or boolean).
- If multiple active rows map to the same assay, keep the one with the latest `effective_month` (string comparison is fine for YYYY-MM format).
- Extract `payment_per_run_per_lab_usd`.

### 2d. Resolve active lab count
- From `lab_overrides.csv`, keep rows where `status` == `approved`.
- For each `assay_id`, keep the row with the highest `revision`.
- Use the `active_labs` from that row.
- If no approved override exists for an in-scope assay, use `default_active_labs` from the manifest.

### 2e. Resolve carrier cost
- Match each assay's `carrier_type` to `carrier_cost.csv` to get `carrier_cost_usd`.
- **IMPORTANT**: Read the test file to determine how `annual_carrier_cost` is computed. The carrier cost line item likely uses: `carrier_cost_usd * active_labs * runs_per_year` OR it might be `carrier_cost_usd * runs_per_year` (per-shipment). Check the test expectations to determine the correct formula. If the test file has expected values, back-calculate to confirm the formula.

### 2f. Compute per-assay financials
- Small-kit: 24 runs/year, `tests_per_lab_per_run_small`
- Bulk-kit: 12 runs/year, `tests_per_lab_per_run_bulk`
- `annual_revenue = payment_per_run_per_lab_usd * active_labs * runs_per_year`
- `annual_reagent_cost = reagent_price_per_1000_tests_usd * active_labs * tests_per_lab_per_run * runs_per_year / 1000`
- `annual_carrier_cost` — determine from test expectations (likely `carrier_cost_usd * runs_per_year` per assay, not multiplied by labs, OR `carrier_cost_usd * active_labs * runs_per_year`). Back-calculate from test expected values if available.
- `annual_margin = annual_revenue - annual_reagent_cost - annual_carrier_cost`
- `difference = bulk_margin - small_margin`
- Round all currency to 2 decimals.

### 2g. Compute totals
- Sum margins and differences across all in-scope assays.
- `absolute_total_margin_difference_usd = abs(total_difference)`, rounded to 2 decimals.

### 2h. Decision
- If `abs(total_difference) < 7000`: `adopt_bulk_kit`
- Otherwise: `keep_small_kit`

### 2i. Build the assumptions object
**CRITICAL**: The test expects exactly 8 keys in `assumptions`. Read the test file to find the exact keys and their expected values. Include all of them. The known keys from the task spec are:
- `runs_per_year_small_kit`: 24
- `runs_per_year_bulk_kit`: 12
- `switch_threshold_usd`: 7000
- `lab_override_rule`: exact string from test
- `billing_rule`: exact string from test

Plus 3 more keys the test expects (likely `annual_margin_formula`, `carrier_cost_interpretation`, `decision_rule`). Copy the exact string values from the test assertions.

### 2j. Build output JSON
- Preserve `metadata` from `report_template.json` exactly.
- Sort `assays` by `assay_id` ascending.
- Write to `/root/reagent_policy_report.json` with `json.dump(..., indent=2)`.

### 2k. Build summary markdown
- Write `/root/reagent_policy_summary.md` with 4-8 non-empty lines.
- Include total small-kit margin, total bulk-kit margin, absolute difference, and the decision slug (`adopt_bulk_kit` or `keep_small_kit`).

## 3. Run the script

```bash
cd /root && python solve.py
```

## 4. Validate outputs

```bash
cat /root/reagent_policy_report.json
cat /root/reagent_policy_summary.md
```

Verify the JSON is valid, has the correct structure, correct number of assumption keys, metadata preserved, assays sorted.

## 5. Run the test suite

```bash
cd / && python -m pytest tests/test_outputs.py -v
```

If any test fails, read the error message carefully, fix the issue in `solve.py`, re-run, and re-test. Pay particular attention to:
- Exact string matches in assumptions
- Numerical precision (use `round(x, 2)`)
- The carrier cost formula (back-calculate from expected values if needed)
- The number of keys in assumptions (must be exactly 8)
- Field names matching exactly

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