# Task Instruction

Execute the following steps in order to produce the two required output files.

## Step 1 – Inspect all input files

Read and display the full contents of every input file so you understand their exact structure:
```
cat /root/panel_manifest.json
cat /root/shipper_cost.csv
cat /root/contract_terms.csv
cat /root/network_adjustments.csv
cat /root/lab_capacity_overrides.csv
cat /root/holdouts.json
cat /root/report_template.json
```

## Step 2 – Inspect the verifier

```
cat /root/test_outputs.py
```

Read the test file carefully to understand every assertion, key name, tolerance, and expected value.

## Step 3 – Write a Python script `/root/solve.py` that does all of the following

### 3a – Load data
- Load `panel_manifest.json`, `holdouts.json`, `report_template.json` as JSON.
- Load `shipper_cost.csv`, `contract_terms.csv`, `network_adjustments.csv`, `lab_capacity_overrides.csv` as CSV (use `csv.DictReader`).

### 3b – Determine retained panels
- From `panel_manifest.json`, keep only entries where `analysis_mode == "review"`.
- From `holdouts.json`, build a set of `panel_code` values where `holdout_state == "exclude"`. Remove those panels.

### 3c – Resolve contract terms
For each retained panel:
- Collect all rows from `contract_terms.csv` where `panel_ref` matches either the panel's `panel_name` OR any entry in the panel's `alias_labels` list.
- Keep only rows with `status_flag == "current"`.
- If multiple rows remain, keep the one with the latest `effective_week` (compare as strings if ISO-formatted, or parse as dates).
- Extract `base_payment_per_run_per_lab_usd` (float).

### 3d – Resolve network adjustment
Build a dict from `network_adjustments.csv` mapping `network_tier` → `network_adjustment_per_run_per_lab_usd` (float). For each panel, look up its `network_tier`. If missing, use 0.0.

### 3e – Resolve shipper cost
Build a dict from `shipper_cost.csv` mapping `shipper_class` → `shipper_cost_usd` (float). Look up each panel's `shipper_class`.

### 3f – Resolve active labs
From `lab_capacity_overrides.csv`:
- Keep only rows where `approval == "approved"`.
- Discard rows where `rev` is blank/empty or `active_labs` is blank/empty.
- Group by `panel_code`. For each group, keep the row with the highest numeric `rev`.
- Build a dict `panel_code → active_labs` (int).
- For any retained panel not in this dict, use `default_active_labs` from `panel_manifest.json`.

### 3g – Compute per-panel financials
For each retained panel, compute:
```
total_payment_per_run_per_lab_usd = base_payment_per_run_per_lab_usd + network_adjustment_per_run_per_lab_usd

# 14-day
annual_revenue_14 = total_payment_per_run_per_lab_usd * active_labs * 26
annual_reagent_cost_14 = reagent_cost_per_1000_tests_usd * active_labs * tests_per_lab_per_run_14_day * 26 / 1000
annual_shipper_cost_14 = shipper_cost_usd * 26
annual_margin_14 = annual_revenue_14 - annual_reagent_cost_14 - annual_shipper_cost_14

# 28-day (same formulas but 13 runs, tests_per_lab_per_run_28_day)
annual_revenue_28 = total_payment_per_run_per_lab_usd * active_labs * 13
annual_reagent_cost_28 = reagent_cost_per_1000_tests_usd * active_labs * tests_per_lab_per_run_28_day * 13 / 1000
annual_shipper_cost_28 = shipper_cost_usd * 13
annual_margin_28 = annual_revenue_28 - annual_reagent_cost_28 - annual_shipper_cost_28

annual_margin_difference_28_minus_14 = annual_margin_28 - annual_margin_14
```

Note: `shipper_cost_usd` is per-shipment, so annual shipper cost = `shipper_cost_usd * runs_per_year`. (Verify this interpretation against the data and test expectations. If the verifier seems to expect `shipper_cost_usd * active_labs * runs_per_year`, adjust accordingly. Check the test file for hints.)

### 3h – Build panel output objects
Each panel object in the `analysis.panels` list must have EXACTLY these keys (no more, no less):
```
panel_code, panel_name, active_labs, reagent_cost_per_1000_tests_usd,
network_tier, network_adjustment_per_run_per_lab_usd, shipper_class,
shipper_cost_usd, base_payment_per_run_per_lab_usd,
total_payment_per_run_per_lab_usd, tests_per_lab_per_run_14_day,
tests_per_lab_per_run_28_day, annual_reagent_cost_14_day_usd,
annual_reagent_cost_28_day_usd, annual_shipper_cost_14_day_usd,
annual_shipper_cost_28_day_usd, annual_revenue_14_day_usd,
annual_revenue_28_day_usd, annual_margin_14_day_usd,
annual_margin_28_day_usd, annual_margin_difference_28_minus_14_usd
```
All currency values rounded to 2 decimal places. Sort panels by `panel_code` ascending.

### 3i – Compute totals
```
total_annual_margin_14_day_usd = sum of all panels' annual_margin_14_day_usd (rounded to 2)
total_annual_margin_28_day_usd = sum of all panels' annual_margin_28_day_usd (rounded to 2)
total_annual_margin_difference_28_minus_14_usd = total_annual_margin_28_day_usd - total_annual_margin_14_day_usd (rounded to 2)
absolute_total_margin_difference_usd = abs(total_annual_margin_difference_28_minus_14_usd) (rounded to 2)
```

### 3j – Decision
- If `absolute_total_margin_difference_usd < 6000`: decision = `"adopt_28_day"`
- Else: decision = `"keep_14_day"`
- Include a justification string that mentions the absolute difference and the threshold.

### 3k – Assemble the JSON report
The `assumptions` block must have EXACTLY these keys:
```json
{
  "runs_per_year_14_day": 26,
  "runs_per_year_28_day": 13,
  "switch_threshold_usd": 6000,
  "override_rule": "highest numeric approved rev with non-empty active_labs, else default_active_labs",
  "holdout_rule": "exclude holdout_state=exclude",
  "adjustment_rule": "missing network_tier adjustment defaults to 0.0"
}
```

Copy `metadata` and `audit_notes` from `report_template.json` exactly as-is (preserve every field and value).

Write the final JSON to `/root/diagpanel_policy_report.json` with `json.dump(..., indent=2)`.

### 3l – Write the Markdown summary
Write `/root/diagpanel_policy_summary.md` with 4-8 non-empty lines including:
- Total 14-day margin (USD)
- Total 28-day margin (USD)
- Absolute difference (USD)
- Final decision slug (`adopt_28_day` or `keep_14_day`)

Example format:
```
# Diagnostics Panel Policy Summary

Total 14-day annual margin: $X.XX USD
Total 28-day annual margin: $Y.YY USD
Absolute margin difference: $Z.ZZ USD
Recommendation: adopt_28_day
```

## Step 4 – Run the script
```
cd /root && python solve.py
```

## Step 5 – Validate outputs
```
cat /root/diagpanel_policy_report.json
cat /root/diagpanel_policy_summary.md
```
Manually verify:
- `metadata` and `audit_notes` match `report_template.json` exactly.
- `assumptions` has exactly the 6 required keys.
- Each panel object has exactly the 21 required keys.
- `totals` has exactly the 4 required keys.
- All currency values have 2 decimal places.
- Panels are sorted by `panel_code`.

## Step 6 – Run the verifier
```
cd /root && python -m pytest test_outputs.py -v
```

If any test fails, read the error carefully, fix the issue in `solve.py`, re-run, and repeat until all tests pass.

## CRITICAL NOTES from previous failure:
- The assumptions block MUST use the exact keys: `runs_per_year_14_day`, `runs_per_year_28_day`, `switch_threshold_usd`, `override_rule`, `holdout_rule`, `adjustment_rule`. Do NOT add extra keys like `decision_threshold_usd` or `shipper_cost_treatment`.
- Panel objects MUST use `annual_margin_difference_28_minus_14_usd` (NOT `difference_28_vs_14_day_usd`).
- Panel objects MUST include `reagent_cost_per_1000_tests_usd`, `tests_per_lab_per_run_14_day`, `tests_per_lab_per_run_28_day`.
- Totals MUST use `total_annual_margin_14_day_usd`, `total_annual_margin_28_day_usd`, `total_annual_margin_difference_28_minus_14_usd`, `absolute_total_margin_difference_usd`.
- Match every key name character-for-character with the schema in the task description.

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