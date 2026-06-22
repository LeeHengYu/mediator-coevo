# Task Instruction

Implement the full solution as a Python script `/root/solve.py` that reads all input files, performs the cost-benefit analysis, and writes both output files. Follow these steps precisely:

## Step 1: Inspect all input files
Before writing any code, read and display the contents of every input file:
```
cat /root/panel_manifest.json
cat /root/shipper_cost.csv
cat /root/contract_terms.csv
cat /root/network_adjustments.csv
cat /root/lab_capacity_overrides.csv
cat /root/holdouts.json
cat /root/report_template.json
```

Also inspect the test file to understand the exact verifier expectations:
```
find /root -name '*.py' -path '*/test*' | head -20
```
Then cat any test files found (e.g., `cat /tests/test_outputs.py` or similar).

## Step 2: Write `/root/solve.py`

Create a Python script that does the following:

### 2a. Load data
- Load `panel_manifest.json` → list of panel objects
- Load `shipper_cost.csv` → dict mapping `shipper_class` → `shipper_cost_usd` (float)
- Load `contract_terms.csv` → list of contract rows
- Load `network_adjustments.csv` → dict mapping `network_tier` → `network_adjustment_per_run_per_lab_usd` (float)
- Load `lab_capacity_overrides.csv` → list of override rows
- Load `holdouts.json` → list of holdout objects
- Load `report_template.json` → template dict

### 2b. Filter panels
- Keep only panels where `analysis_mode` == `"review"`
- Exclude any panel whose `panel_code` appears in holdouts with `holdout_state` == `"exclude"`
- These are the "retained" panels.

### 2c. Resolve contract terms
For each retained panel:
- Find all rows in `contract_terms.csv` where `panel_ref` matches either the panel's `panel_name` OR any entry in the panel's `alias_labels` list.
- Keep only rows where `status_flag` == `"current"`.
- If multiple current rows match, keep the one with the latest `effective_week` (string comparison should work if format is consistent, but parse as date if needed).
- Extract `base_payment_per_run_per_lab_usd` from the matched contract row.

### 2d. Resolve network adjustment
- Look up the panel's `network_tier` in `network_adjustments.csv`.
- If not found, use `0.0`.

### 2e. Resolve active labs
- From `lab_capacity_overrides.csv`, find rows matching the panel's `panel_code`.
- Keep only rows where `approval` == `"approved"`.
- Discard rows where `rev` is blank/empty or `active_labs` is blank/empty.
- Among remaining rows, keep the one with the highest numeric `rev`.
- Use its `active_labs` (as integer).
- If no valid row exists, use `default_active_labs` from the panel manifest.

### 2f. Resolve shipper cost
- Look up the panel's `shipper_class` in `shipper_cost.csv` to get `shipper_cost_usd`.

### 2g. Compute per-panel metrics
For each retained panel, compute:
- `total_payment_per_run_per_lab_usd` = `base_payment_per_run_per_lab_usd` + `network_adjustment_per_run_per_lab_usd`
- **14-day model** (26 runs/year):
  - `annual_revenue_14_day_usd` = `total_payment_per_run_per_lab_usd * active_labs * 26`
  - `annual_reagent_cost_14_day_usd` = `reagent_cost_per_1000_tests_usd * active_labs * tests_per_lab_per_run_14_day * 26 / 1000`
  - `annual_shipper_cost_14_day_usd` = `shipper_cost_usd * 26`
  - `annual_margin_14_day_usd` = revenue - reagent_cost - shipper_cost
- **28-day model** (13 runs/year): same formulas with 13 runs and `tests_per_lab_per_run_28_day`
  - `annual_shipper_cost_28_day_usd` = `shipper_cost_usd * 13`
- `annual_margin_difference_28_minus_14_usd` = margin_28 - margin_14
- Round ALL currency values to 2 decimal places.

### 2h. Build output panel objects
Each panel object in `analysis.panels` must have EXACTLY these keys (in this order conceptually, but JSON key order doesn't matter for correctness):
- `panel_code`, `panel_name`, `active_labs`, `reagent_cost_per_1000_tests_usd`, `network_tier`, `network_adjustment_per_run_per_lab_usd`, `shipper_class`, `shipper_cost_usd`, `base_payment_per_run_per_lab_usd`, `total_payment_per_run_per_lab_usd`, `tests_per_lab_per_run_14_day`, `tests_per_lab_per_run_28_day`, `annual_reagent_cost_14_day_usd`, `annual_reagent_cost_28_day_usd`, `annual_shipper_cost_14_day_usd`, `annual_shipper_cost_28_day_usd`, `annual_revenue_14_day_usd`, `annual_revenue_28_day_usd`, `annual_margin_14_day_usd`, `annual_margin_28_day_usd`, `annual_margin_difference_28_minus_14_usd`

Sort panels by `panel_code` ascending.

### 2i. Compute totals
- `total_annual_margin_14_day_usd` = sum of all panels' `annual_margin_14_day_usd`
- `total_annual_margin_28_day_usd` = sum of all panels' `annual_margin_28_day_usd`
- `total_annual_margin_difference_28_minus_14_usd` = sum of all panels' `annual_margin_difference_28_minus_14_usd`
- `absolute_total_margin_difference_usd` = `abs(total_annual_margin_difference_28_minus_14_usd)`
- Round all to 2 decimals.

### 2j. Decision
- If `absolute_total_margin_difference_usd < 6000`: decision = `"adopt_28_day"`
- Otherwise: decision = `"keep_14_day"`
- Justification: a short string explaining the decision referencing the threshold and the absolute difference.

### 2k. Build assumptions dict
The `analysis.assumptions` dict must contain EXACTLY:
```python
{
    "runs_per_year_14_day": 26,
    "runs_per_year_28_day": 13,
    "switch_threshold_usd": 6000,
    "override_rule": "highest numeric approved rev with non-empty active_labs, else default_active_labs",
    "holdout_rule": "exclude holdout_state=exclude",
    "adjustment_rule": "missing network_tier adjustment defaults to 0.0"
}
```

### 2l. Assemble final JSON
- Copy `metadata` and `audit_notes` from `report_template.json` exactly as-is.
- Add the `analysis` block with `assumptions`, `panels`, `totals`, `recommendation`.
- Write to `/root/diagpanel_policy_report.json` with `json.dump(..., indent=2)`.

### 2m. Write summary markdown
Write `/root/diagpanel_policy_summary.md` with 4-8 non-empty lines including:
- Total 14-day margin (USD) with the numeric value
- Total 28-day margin (USD) with the numeric value
- Absolute difference (USD) with the numeric value
- Final decision using the exact slug (`adopt_28_day` or `keep_14_day`)

## Step 3: Run the script
```
cd /root && python solve.py
```

## Step 4: Validate output
- `cat /root/diagpanel_policy_report.json` and verify:
  - `metadata` and `audit_notes` are present and match template
  - `analysis.assumptions` has all 6 keys
  - `analysis.panels` is sorted by panel_code, each panel has all required `_usd` suffixed keys
  - `analysis.totals` has all 4 required keys
  - `analysis.recommendation` has `decision` and `justification`
- `cat /root/diagpanel_policy_summary.md` and verify 4-8 non-empty lines with required content

## Step 5: Run the test suite
Find and run the test file:
```
python -m pytest /tests/ -v 2>&1 || python -m pytest /root/tests/ -v 2>&1
```
If tests fail, read the error messages carefully, fix the issue in `solve.py`, re-run, and re-test. Pay special attention to:
- Exact key names (the `_usd` and `_day_usd` suffixes are critical)
- The `assumptions` dict must not be empty
- All totals keys must exist
- Currency values must be rounded to 2 decimals
- Panel sort order must be by `panel_code` ascending

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