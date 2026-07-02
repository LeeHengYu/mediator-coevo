# Task Instruction

Execute the following steps in order to produce the two required output files.

## Step 1 – Inspect all input files

Read and display the full contents of every input file:

```
cat /root/panel_manifest.json
cat /root/shipper_cost.csv
cat /root/contract_terms.csv
cat /root/network_adjustments.csv
cat /root/lab_capacity_overrides.csv
cat /root/holdouts.json
cat /root/report_template.json
```

## Step 2 – Write and run a Python script

Create `/root/solve.py` that does all of the following (read the rules below carefully):

### 2a – Load data
- Load `panel_manifest.json` (list of panel objects).
- Load `shipper_cost.csv`, `contract_terms.csv`, `network_adjustments.csv`, `lab_capacity_overrides.csv` (use csv.DictReader).
- Load `holdouts.json` (list of holdout objects).
- Load `report_template.json`.

### 2b – Determine retained panels
- From `panel_manifest.json`, keep only panels where `analysis_mode` == `"review"`.
- Build a set of excluded panel_codes from `holdouts.json` where `holdout_state` == `"exclude"`.
- Remove any panel whose `panel_code` is in the excluded set.
- The remaining panels are the "retained" panels.

### 2c – Resolve contract terms
For each retained panel:
- Match `contract_terms.csv` rows where `panel_ref` equals the panel's `panel_name` OR `panel_ref` appears in the panel's `alias_labels` list.
- Keep only rows where `status_flag` == `"current"`.
- If multiple rows match, keep the one with the latest `effective_week` (compare as strings – they should be ISO-like dates or week identifiers; if purely numeric, compare numerically).
- Extract `base_payment_per_run_per_lab_usd` (float).

### 2d – Network adjustments
- Build a dict from `network_adjustments.csv` mapping `network_tier` → `network_adjustment_per_run_per_lab_usd` (float).
- For each retained panel, look up its `network_tier`. If not found, use 0.0.

### 2e – Active labs (lab_capacity_overrides)
- Filter rows: `approval` == `"approved"`, `rev` is not blank/empty, `active_labs` is not blank/empty.
- Group by `panel_code`. For each group, keep the row with the highest numeric `rev`.
- For each retained panel, if there is a matching override row, use its `active_labs` (int). Otherwise use `default_active_labs` from the manifest.

### 2f – Shipper cost
- Build a dict from `shipper_cost.csv` mapping `shipper_class` → `shipper_cost_usd` (float).
- For each retained panel, look up `shipper_class` from the manifest and get the cost.

### 2g – Compute per-panel numbers
For each retained panel:
- `total_payment_per_run_per_lab_usd` = `base_payment_per_run_per_lab_usd` + `network_adjustment_per_run_per_lab_usd`
- 14-day model: `runs_per_year` = 26, `tests_per_lab_per_run` = `tests_per_lab_per_run_14_day`
- 28-day model: `runs_per_year` = 13, `tests_per_lab_per_run` = `tests_per_lab_per_run_28_day`
- `annual_revenue = total_payment_per_run_per_lab_usd * active_labs * runs_per_year`
- `annual_reagent_cost = reagent_cost_per_1000_tests_usd * active_labs * tests_per_lab_per_run * runs_per_year / 1000`
- `annual_shipper_cost = shipper_cost_usd * runs_per_year`  (NOTE: the shipper cost is per shipment/run, not per lab – but re-check the data; if shipper_cost.csv has a column that suggests per-lab, adjust accordingly. The instruction says "annual shipper cost" without an explicit formula with active_labs, so use `shipper_cost_usd * runs_per_year` unless the data clearly indicates otherwise. Actually, looking at the margin formula: `annual_margin = annual_revenue - annual_reagent_cost - annual_shipper_cost`, and there is no explicit formula given for annual_shipper_cost. The shipper cost is matched by shipper_class, so it is likely a per-run cost. Use `shipper_cost_usd * runs_per_year`.)
- `annual_margin = annual_revenue - annual_reagent_cost - annual_shipper_cost`
- `annual_margin_difference_28_minus_14 = annual_margin_28 - annual_margin_14`
- Round all currency values to 2 decimal places.

### 2h – Totals and decision
- `total_annual_margin_14_day_usd` = sum of all per-panel 14-day margins
- `total_annual_margin_28_day_usd` = sum of all per-panel 28-day margins
- `total_annual_margin_difference_28_minus_14_usd` = sum of all per-panel differences
- `absolute_total_margin_difference_usd` = abs(total_difference)
- Round all to 2 decimals.
- Decision: if `absolute_total_margin_difference_usd < 6000` → `"adopt_28_day"`, else `"keep_14_day"`.
- Justification: a short human-readable string explaining the decision (e.g., "Absolute margin difference of $X is below/above the $6,000 threshold; recommend adopt_28_day/keep_14_day.").

### 2i – Build JSON output
- Start from `report_template.json`. Preserve `metadata` and `audit_notes` exactly as they appear in the template.
- Set `analysis.assumptions` to the exact dict shown in the schema (with the exact keys: `runs_per_year_14_day`, `runs_per_year_28_day`, `switch_threshold_usd`, `override_rule`, `holdout_rule`, `adjustment_rule` with the exact string values from the schema).
- Set `analysis.panels` sorted by `panel_code` ascending.
- Set `analysis.totals` and `analysis.recommendation`.
- Write to `/root/diagpanel_policy_report.json` with `json.dump(..., indent=2)`.

### 2j – Build markdown summary
Write `/root/diagpanel_policy_summary.md` with 4-8 non-empty lines including:
- Total 14-day annual margin with comma-formatted USD (e.g., `$12,345.67`)
- Total 28-day annual margin with comma-formatted USD
- Absolute margin difference with comma-formatted USD
- Final decision using the exact slug `adopt_28_day` or `keep_14_day`

Use Python's `"{:,.2f}".format(value)` for comma-formatted currency in the summary.

## Step 3 – Run the script

```bash
cd /root && python solve.py
```

## Step 4 – Validate outputs

```bash
cat /root/diagpanel_policy_report.json
cat /root/diagpanel_policy_summary.md
python -c "import json; d=json.load(open('/root/diagpanel_policy_report.json')); print('Keys:', list(d.keys())); print('Panels:', len(d['analysis']['panels'])); print('Totals:', d['analysis']['totals']); print('Decision:', d['analysis']['recommendation'])"
```

Verify:
- JSON is valid and parseable.
- `metadata` and `audit_notes` match the template exactly.
- `assumptions` has exactly the 6 keys specified.
- `panels` are sorted by `panel_code`.
- All currency values have exactly 2 decimal places.
- Summary has 4-8 non-empty lines, includes comma-formatted USD values and the exact decision slug.
- No extra or missing keys in the schema.

## Step 5 – Run verifier if available

```bash
ls /root/test_output.py 2>/dev/null && cd /root && python -m pytest test_output.py -v
```

If any test fails, read the error carefully, fix the issue in solve.py, and re-run. Pay special attention to:
- Missing or extra keys in `assumptions`
- Numeric formatting in the summary (use comma-separated like `1,234.56`, not plain `1234.56`)
- The `justification` key must be present in `recommendation`
- Shipper cost formula – if tests fail, try `shipper_cost_usd * active_labs * runs_per_year` instead of `shipper_cost_usd * runs_per_year` and re-check.

Iterate until all tests pass.

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