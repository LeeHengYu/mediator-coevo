# Task Instruction

## Task: Healthcare Cost-Benefit Analysis – diagpanel_14v28

You must produce two output files by reading and processing the input data files according to precise rules. Follow every step carefully.

### Step 1: Read all input files

```
cat /root/panel_manifest.json
cat /root/shipper_cost.csv
cat /root/contract_terms.csv
cat /root/network_adjustments.csv
cat /root/lab_capacity_overrides.csv
cat /root/holdouts.json
cat /root/report_template.json
```

Inspect them all before writing any code.

### Step 2: Write a Python script `/root/solve.py` that does the following

#### 2a: Load data
- Load `panel_manifest.json`, `holdouts.json`, `report_template.json` as JSON.
- Load `shipper_cost.csv`, `contract_terms.csv`, `network_adjustments.csv`, `lab_capacity_overrides.csv` as CSV (use `csv.DictReader`).

#### 2b: Filter panels
- From `panel_manifest.json`, keep only panels where `analysis_mode` == `"review"`.
- From `holdouts.json`, exclude any panel whose `panel_code` appears with `holdout_state` == `"exclude"`.
- The remaining panels are "retained panels".

#### 2c: Resolve contract terms
- For each retained panel, find matching rows in `contract_terms.csv` where `panel_ref` matches either the panel's `panel_name` OR any entry in its `alias_labels` list.
- Keep only rows where `status_flag` == `"current"`.
- If multiple current rows match the same panel, keep the one with the latest `effective_week` (compare as strings if they are ISO date-like, or parse appropriately).
- Extract `base_payment_per_run_per_lab_usd` (float).

#### 2d: Network adjustments
- Build a lookup from `network_adjustments.csv` keyed by `network_tier`.
- For each retained panel, look up `network_adjustment_per_run_per_lab_usd` by the panel's `network_tier`.
- If the tier is not found, use `0.0`.

#### 2e: Active labs
- From `lab_capacity_overrides.csv`, keep only rows where `approval` == `"approved"`.
- Discard rows where `rev` is blank/empty or `active_labs` is blank/empty.
- Among remaining valid rows for the same `panel_code`, keep the one with the highest numeric `rev` (convert `rev` to int or float for comparison).
- If no valid override row exists for a retained panel, use `default_active_labs` from `panel_manifest.json`.

#### 2f: Shipper cost
- Build a lookup from `shipper_cost.csv` keyed by `shipper_class`.
- For each panel, look up `shipper_cost_usd` by the panel's `shipper_class`.

#### 2g: Compute per-panel financials
For each retained panel:
- `total_payment_per_run_per_lab_usd` = `base_payment_per_run_per_lab_usd` + `network_adjustment_per_run_per_lab_usd`
- 14-day model: `runs_per_year` = 26, `tests_per_lab_per_run` = `tests_per_lab_per_run_14_day`
- 28-day model: `runs_per_year` = 13, `tests_per_lab_per_run` = `tests_per_lab_per_run_28_day`
- `annual_revenue = total_payment_per_run_per_lab_usd * active_labs * runs_per_year`
- `annual_reagent_cost = reagent_cost_per_1000_tests_usd * active_labs * tests_per_lab_per_run * runs_per_year / 1000`
- `annual_shipper_cost = shipper_cost_usd * runs_per_year`  (shipper cost is per shipment, i.e., per run — confirm from data; if shipper_cost.csv has a single cost per shipper_class, it's the cost per shipment)
  - **IMPORTANT**: Re-check the data. The shipper cost might be annual or per-run. Look at the CSV column names and values to determine this. If the column is just `shipper_cost_usd` with no qualifier, treat it as **per run** (i.e., `annual_shipper_cost = shipper_cost_usd * runs_per_year`). But if values look annual already, use them directly. Use your judgment from the data magnitudes.
- `annual_margin = annual_revenue - annual_reagent_cost - annual_shipper_cost`
- `annual_margin_difference_28_minus_14 = annual_margin_28_day - annual_margin_14_day`

Round ALL currency values to 2 decimal places.

#### 2h: Totals
- `total_annual_margin_14_day_usd` = sum of all panels' `annual_margin_14_day_usd`
- `total_annual_margin_28_day_usd` = sum of all panels' `annual_margin_28_day_usd`
- `total_annual_margin_difference_28_minus_14_usd` = sum of all panels' `annual_margin_difference_28_minus_14_usd`
- `absolute_total_margin_difference_usd` = `abs(total_annual_margin_difference_28_minus_14_usd)`
- Round all to 2 decimals.

#### 2i: Decision
- If `absolute_total_margin_difference_usd < 6000`, decision = `"adopt_28_day"`
- Otherwise, decision = `"keep_14_day"`
- Justification: a brief string explaining the numbers and threshold.

#### 2j: Build JSON output
- Start with `metadata` and `audit_notes` copied **exactly** from `report_template.json`.
- Build the `analysis` object with `assumptions`, `panels` (sorted by `panel_code` ascending), `totals`, and `recommendation`.
- The `assumptions` object must have exactly these flat keys:
  - `runs_per_year_14_day`: 26
  - `runs_per_year_28_day`: 13
  - `switch_threshold_usd`: 6000
  - `override_rule`: `"highest numeric approved rev with non-empty active_labs, else default_active_labs"`
  - `holdout_rule`: `"exclude holdout_state=exclude"`
  - `adjustment_rule`: `"missing network_tier adjustment defaults to 0.0"`
- Each panel object must have exactly the flat keys shown in the schema (no nested sub-objects).
- Write to `/root/diagpanel_policy_report.json` with `json.dump(..., indent=2)`.

#### 2k: Build markdown summary
- Write `/root/diagpanel_policy_summary.md` with 4-8 non-empty lines.
- **CRITICAL**: Format all currency values WITH comma separators, e.g., `f'{value:,.2f}'`. The test expects strings like `"1,090.76"` not `"1090.76"`.
- Must include:
  - Total 14-day margin (USD) with commas
  - Total 28-day margin (USD) with commas
  - Absolute difference (USD) with commas
  - Final decision using the exact slug (`adopt_28_day` or `keep_14_day`)

Example lines:
```
# Diagpanel Policy Summary

Total 14-day annual margin: $1,090.76
Total 28-day annual margin: $-2,092.94
Absolute margin difference: $3,183.70
Final decision: adopt_28_day
```

Note: For negative numbers, `f'{value:,.2f}'` produces e.g. `-2,092.94`. The test should find the number with commas in the text. Make sure the comma-formatted number appears in the line.

### Step 3: Run the script
```bash
cd /root && python solve.py
```

### Step 4: Validate outputs
```bash
cat /root/diagpanel_policy_report.json | python -m json.tool > /dev/null && echo 'JSON valid'
cat /root/diagpanel_policy_summary.md
```

Verify:
- JSON is valid and parseable
- JSON has `metadata`, `audit_notes`, `analysis` with `assumptions`, `panels`, `totals`, `recommendation`
- `panels` are sorted by `panel_code`
- All currency values have 2 decimal places
- Summary has comma-formatted numbers
- Summary contains the decision slug

### Step 5: Run the verifier
```bash
cd /root && python -m pytest test_output.py -v
```

If any test fails, read the error carefully, fix the issue in `solve.py`, re-run, and re-verify. Pay special attention to:
- Schema/key mismatches (flat keys, not nested)
- Comma formatting in the markdown summary
- Correct matching of contract_terms via panel_name AND alias_labels
- Shipper cost calculation (per-run vs annual)
- Sorting order of panels

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