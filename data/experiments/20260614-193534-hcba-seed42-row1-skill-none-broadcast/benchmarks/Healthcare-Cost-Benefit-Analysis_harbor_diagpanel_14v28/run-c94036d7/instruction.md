# Task Instruction

Execute the following steps carefully to produce `/root/diagpanel_policy_report.json` and `/root/diagpanel_policy_summary.md`.

## Step 0 — Inspect all input files and the test file

Read every input file in full:
```
cat /root/panel_manifest.json
cat /root/shipper_cost.csv
cat /root/contract_terms.csv
cat /root/network_adjustments.csv
cat /root/lab_capacity_overrides.csv
cat /root/holdouts.json
cat /root/report_template.json
```
Also read the verifier:
```
cat /root/tests/test_outputs.py
```
Understand exactly what the verifier checks (field names, tolerances, ordering, formatting of currency with commas, etc.) before writing any code.

## Step 1 — Write a Python script `/root/solve.py`

After reading all files, write a single Python script that:

### 1a. Load data
- Load `panel_manifest.json`, `holdouts.json`, `report_template.json` as JSON.
- Load `shipper_cost.csv`, `contract_terms.csv`, `network_adjustments.csv`, `lab_capacity_overrides.csv` as CSV (use the `csv` module with DictReader).

### 1b. Determine retained panels
- From `panel_manifest.json`, keep only entries where `analysis_mode == "review"`.
- Remove any panel whose `panel_code` appears in `holdouts.json` with `holdout_state == "exclude"`.

### 1c. Match contract terms — BE VERY CAREFUL HERE
- For each retained panel, find rows in `contract_terms.csv` where `panel_ref` matches either the panel's `panel_name` OR any string in the panel's `alias_labels` list.
- Keep only rows where `status_flag == "current"`.
- If multiple current rows match the same panel, keep the one with the latest `effective_week` (compare as strings if they are ISO dates, or parse them).
- Extract `base_payment_per_run_per_lab_usd` from the winning row.

### 1d. Network adjustment
- Build a lookup from `network_adjustments.csv` keyed by `network_tier`.
- For each panel, look up `network_adjustment_per_run_per_lab_usd` by the panel's `network_tier`. Default to `0.0` if not found.
- `total_payment_per_run_per_lab_usd = base_payment + network_adjustment`

### 1e. Active labs
- From `lab_capacity_overrides.csv`, keep rows where `approval == "approved"` AND `rev` is not blank AND `active_labs` is not blank.
- Among valid rows for the same `panel_code`, keep the one with the highest numeric `rev`.
- If no valid override row exists for a panel, use `default_active_labs` from the manifest.

### 1f. Shipper cost
- Build a lookup from `shipper_cost.csv` keyed by `shipper_class`.
- For each panel, look up `shipper_cost_usd` by the panel's `shipper_class`.
- **IMPORTANT**: The shipper cost is a per-shipment cost. The annual shipper cost is `shipper_cost_usd * runs_per_year` (NOT multiplied by active_labs, NOT multiplied by tests). Verify this interpretation against the test expectations. If the test file reveals a different formula, use that.
- Actually, re-read the task: it says `annual_shipper_cost` but doesn't give an explicit formula for it. Check the test file for clues. The previous run had a huge discrepancy (26185.96 vs 1090.76 for total 14-day margin), which suggests shipper cost was being over-counted (e.g., multiplied by active_labs when it shouldn't be, or vice versa).
- **Decision**: Start by computing `annual_shipper_cost = shipper_cost_usd * runs_per_year` (per panel, not per lab). If the test fails, try `shipper_cost_usd * active_labs * runs_per_year`. Read the test file first to see if it reveals the expected values.

### 1g. Compute per-panel figures
For each retained panel, for each cadence (14-day with 26 runs, 28-day with 13 runs):
- `annual_revenue = total_payment_per_run_per_lab_usd * active_labs * runs_per_year`
- `annual_reagent_cost = reagent_cost_per_1000_tests_usd * active_labs * tests_per_lab_per_run * runs_per_year / 1000`
- `annual_shipper_cost = shipper_cost_usd * runs_per_year` (adjust if test file indicates otherwise)
- `annual_margin = annual_revenue - annual_reagent_cost - annual_shipper_cost`
- `annual_margin_difference_28_minus_14 = margin_28 - margin_14`

Round all currency values to 2 decimal places.

### 1h. Totals and decision
- Sum all per-panel `annual_margin_14_day_usd` → `total_annual_margin_14_day_usd`
- Sum all per-panel `annual_margin_28_day_usd` → `total_annual_margin_28_day_usd`
- `total_annual_margin_difference_28_minus_14_usd = total_28 - total_14`
- `absolute_total_margin_difference_usd = abs(total_difference)`
- If `absolute_total_margin_difference_usd < 6000` → `adopt_28_day`, else `keep_14_day`.

### 1i. Build JSON output
- Start from `report_template.json` to preserve `metadata` and `audit_notes` exactly.
- Add the `analysis` block with `assumptions`, `panels` (sorted by `panel_code` ascending), `totals`, and `recommendation`.
- Write to `/root/diagpanel_policy_report.json` with `json.dump(..., indent=2)`.

### 1j. Build markdown summary
- Write `/root/diagpanel_policy_summary.md` with 4–8 non-empty lines.
- Include total 14-day margin, total 28-day margin, absolute difference, and the decision slug.
- **Format currency values with commas** (use `f"{value:,.2f}"` in Python). The avoid artifact from `harbor_syncpack_28v56` shows that missing comma formatting caused a test failure.

## Step 2 — Run the script
```
cd /root && python solve.py
```

## Step 3 — Validate outputs exist and look correct
```
cat /root/diagpanel_policy_report.json
cat /root/diagpanel_policy_summary.md
```
Check that:
- JSON is valid and has all required fields.
- Panels are sorted by `panel_code`.
- `metadata` and `audit_notes` match the template exactly.
- Currency values in the summary have commas.

## Step 4 — Run the verifier
```
cd /root && python -m pytest tests/test_outputs.py -v
```

## Step 5 — If tests fail, debug
- Read the exact assertion error messages.
- Check which specific values are wrong.
- Re-examine the data files and your matching/calculation logic.
- Common pitfalls to check:
  - Is `shipper_cost` annual = per_shipment * runs_per_year, or per_shipment * runs_per_year * active_labs?
  - Are `alias_labels` being checked correctly (they might be a JSON array of strings in the manifest)?
  - Is `effective_week` being compared correctly (string comparison works for ISO dates)?
  - Are `rev` values being compared as numbers (not strings)?
  - Is `active_labs` from the override being parsed as a number?
  - Are `base_payment_per_run_per_lab_usd` and other numeric CSV fields being parsed as floats?
- Fix and re-run until all tests pass.

## Key Warnings from Previous Failure
1. The previous run had `total_annual_margin_14_day_usd = 26185.96` but expected `1090.76`. This is a ~24x difference, strongly suggesting shipper cost was NOT multiplied by active_labs when it should have been, OR base_payment was wrong due to incorrect contract matching.
2. A per-panel value `369.2` was expected to be `6276.4` — again suggesting a multiplicative factor is off.
3. Read the test file FIRST to understand what exact values are expected, then work backwards to verify your logic.

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
Task metadata: author_email=gpt54@example.com, author_name=GPT-5.4, category=financial-analysis, difficulty=medium, tags=[diagnostics, json, csv, template-update, decision-analysis].
Verifier config: timeout_sec=900.0.