# Task Instruction

Execute the following steps in order:

1. **Inspect all input files** to understand their structure:
```bash
cat /root/assay_manifest.json
cat /root/carrier_cost.csv
cat /root/billing.csv
cat /root/lab_overrides.csv
cat /root/report_template.json
```

2. **Inspect the verifier test** to understand exactly what keys, formats, and values are expected:
```bash
cat /root/test_outputs.py
```

3. **Create `/root/solve.py`** that does the following:

**Data Loading:**
- Load `assay_manifest.json`, `carrier_cost.csv`, `billing.csv`, `lab_overrides.csv`, `report_template.json`.

**Filter in-scope assays:**
- From `assay_manifest.json`, keep only entries where `in_scope` is `true`.

**Billing resolution:**
- For each in-scope assay, find billing rows where `assay_label` matches either `assay_name` or any alias in the assay's aliases list.
- Keep only rows where `is_active` is `true` (handle string "true"/"True" or boolean).
- If multiple active rows match, keep the one with the latest `effective_month`.
- Extract `payment_per_run_per_lab_usd` from the retained row.

**Lab overrides:**
- From `lab_overrides.csv`, keep only rows where `status` is `approved`.
- For each in-scope assay's `assay_id`, if there are approved rows, keep the one with the highest `revision`.
- Use its `active_labs` value. If no approved row exists, use `default_active_labs` from the manifest.

**Carrier cost:**
- Match each assay's `carrier_type` to `carrier_cost.csv` to get `carrier_cost_usd`.

**Calculations per assay:**
- `runs_per_year_small = 24`, `runs_per_year_bulk = 12`
- `tests_per_lab_per_run_small` and `tests_per_lab_per_run_bulk` from manifest.
- `reagent_price_per_1000_tests_usd` from manifest.
- `annual_revenue_small = payment_per_run_per_lab_usd * active_labs * 24`
- `annual_revenue_bulk = payment_per_run_per_lab_usd * active_labs * 12`
- `annual_reagent_cost_small = reagent_price_per_1000_tests_usd * active_labs * tests_per_lab_per_run_small * 24 / 1000`
- `annual_reagent_cost_bulk = reagent_price_per_1000_tests_usd * active_labs * tests_per_lab_per_run_bulk * 12 / 1000`
- `annual_carrier_cost_small = carrier_cost_usd * active_labs * 24`
- `annual_carrier_cost_bulk = carrier_cost_usd * active_labs * 12`
- `annual_margin_small = annual_revenue_small - annual_reagent_cost_small - annual_carrier_cost_small`
- `annual_margin_bulk = annual_revenue_bulk - annual_reagent_cost_bulk - annual_carrier_cost_bulk`
- `difference = annual_margin_bulk - annual_margin_small`
- Round ALL currency values to 2 decimal places.

**Per-assay output object** must include ALL of these keys (this is critical — previous run failed because some were missing):
- `assay_id`, `assay_name`, `active_labs`, `reagent_price_per_1000_tests_usd`, `carrier_type`, `carrier_cost_usd`, `payment_per_run_per_lab_usd`, `tests_per_lab_per_run_small`, `tests_per_lab_per_run_bulk`, `annual_reagent_cost_small_kit_usd`, `annual_reagent_cost_bulk_kit_usd`, `annual_carrier_cost_small_kit_usd`, `annual_carrier_cost_bulk_kit_usd`, `annual_revenue_small_kit_usd`, `annual_revenue_bulk_kit_usd`, `annual_margin_small_kit_usd`, `annual_margin_bulk_kit_usd`, `annual_margin_difference_bulk_minus_small_usd`

**Sort** assays by `assay_id` ascending.

**Totals:**
- Sum all per-assay margins for small and bulk.
- `total_difference = total_bulk - total_small`
- `absolute_total = abs(total_difference)`
- Round to 2 decimals.

**Decision:**
- If `absolute_total < 7000`: `adopt_bulk_kit`
- Otherwise: `keep_small_kit`
- Include a justification string.

**Assumptions object** — This is CRITICAL. The previous run failed because the assumptions dict was incomplete. Read the test file carefully to see exactly what keys the verifier expects. The test likely checks for keys beyond the 5 listed in the prompt. After reading `test_outputs.py`, include ALL keys the test expects. At minimum, the prompt specifies these 5:
- `runs_per_year_small_kit`: 24
- `runs_per_year_bulk_kit`: 12
- `switch_threshold_usd`: 7000
- `lab_override_rule`: "highest approved revision per assay_id, else default_active_labs"
- `billing_rule`: "latest active effective_month per assay"

But the test may expect additional keys like `billing_resolution`, `bulk_kit_runs_per_year`, `carrier_cost_formula`, `in_scope_filter`, `small_kit_runs_per_year`, etc. Include whatever the test expects. If the test checks `set(assumptions.keys()) >= {some_set}`, include all keys from that set with appropriate values.

**Metadata:** Copy the `metadata` object from `report_template.json` exactly as-is.

**Write `/root/reagent_policy_report.json`** with proper JSON formatting (indent=2).

**Write `/root/reagent_policy_summary.md`:**
- 4 to 8 non-empty lines.
- Must include total small-kit margin, total bulk-kit margin, absolute difference, and the decision slug.
- **CRITICAL**: Use comma-formatted numbers with `:,.2f` format spec (e.g., `f'{value:,.2f}'`). The previous run failed because it used `:.2f` instead of `:,.2f`. The verifier checks for the comma-formatted string.

4. **Run the solver:**
```bash
cd /root && python solve.py
```

5. **Validate outputs:**
```bash
cat /root/reagent_policy_report.json
cat /root/reagent_policy_summary.md
```

6. **Run the verifier test:**
```bash
cd /root && python -m pytest test_outputs.py -v
```

7. If any test fails, read the error carefully, fix `solve.py` accordingly, and re-run. Pay special attention to:
   - Missing keys in assumptions or assay objects
   - Numerical mismatches (double-check formulas)
   - String formatting in the markdown summary
   - The carrier cost formula: `carrier_cost_usd * active_labs * runs_per_year` (per model)
   - Ensure `is_active` parsing handles both string and boolean types
   - Ensure `revision` comparison is numeric, not string

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