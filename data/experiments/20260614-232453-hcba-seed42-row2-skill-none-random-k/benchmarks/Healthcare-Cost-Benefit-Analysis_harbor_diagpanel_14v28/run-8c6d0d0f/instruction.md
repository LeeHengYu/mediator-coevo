# Task Instruction

Execute the following steps carefully and in order.

## Step 1: Inspect all input files

Read and display the full contents of each input file:
```
cat /root/panel_manifest.json
cat /root/shipper_cost.csv
cat /root/contract_terms.csv
cat /root/network_adjustments.csv
cat /root/lab_capacity_overrides.csv
cat /root/holdouts.json
cat /root/report_template.json
```

Also check if there is a test file:
```
cat /root/test_output.py 2>/dev/null || echo 'No test file found'
```

## Step 2: Write the solution script

After reading ALL files, write a Python script `/root/solve.py` that does the following:

### 2a. Load data
- Load `panel_manifest.json`, `holdouts.json`, `report_template.json` as JSON.
- Load `shipper_cost.csv`, `contract_terms.csv`, `network_adjustments.csv`, `lab_capacity_overrides.csv` as CSV (use pandas or csv module).

### 2b. Filter panels
- From `panel_manifest.json`, keep only panels where `analysis_mode` == `"review"`.
- From `holdouts.json`, exclude any panel whose `panel_code` appears with `holdout_state` == `"exclude"`.
- The remaining panels are the "retained" panels.

### 2c. Resolve contract terms
- For each retained panel, find matching rows in `contract_terms.csv` where `panel_ref` matches either the panel's `panel_name` OR any entry in its `alias_labels` list.
- Keep only rows where `status_flag` == `"current"`.
- If multiple current rows match the same panel, keep the one with the latest `effective_week` (compare as strings or dates, whichever is appropriate — inspect the data format).
- Extract `base_payment_per_run_per_lab_usd` from the winning contract row.

### 2d. Resolve network adjustment
- Match each retained panel's `network_tier` to `network_adjustments.csv`.
- If no match, use `0.0`.
- Extract `network_adjustment_per_run_per_lab_usd`.

### 2e. Resolve active labs
- From `lab_capacity_overrides.csv`, keep only rows where `approval` == `"approved"`.
- Among those, ignore rows where `rev` is blank/empty or `active_labs` is blank/empty.
- If multiple valid rows exist for the same `panel_code`, keep the one with the highest numeric `rev`.
- If no valid override row exists for a retained panel, use `default_active_labs` from `panel_manifest.json`.

### 2f. Resolve shipper cost
- Match each retained panel's `shipper_class` to `shipper_cost.csv` to get `shipper_cost_usd`.

### 2g. Compute per-panel financials
For each retained panel:
- `total_payment_per_run_per_lab_usd` = `base_payment_per_run_per_lab_usd` + `network_adjustment_per_run_per_lab_usd`
- **14-day model** (26 runs/year):
  - `annual_revenue_14_day_usd` = `total_payment_per_run_per_lab_usd * active_labs * 26`
  - `annual_reagent_cost_14_day_usd` = `reagent_cost_per_1000_tests_usd * active_labs * tests_per_lab_per_run_14_day * 26 / 1000`
  - `annual_shipper_cost_14_day_usd` = `shipper_cost_usd * 26`  (shipper cost is per shipment, 26 shipments)
  - `annual_margin_14_day_usd` = revenue - reagent_cost - shipper_cost
- **28-day model** (13 runs/year):
  - Same formulas but with 13 runs and `tests_per_lab_per_run_28_day`
  - `annual_shipper_cost_28_day_usd` = `shipper_cost_usd * 13`
  - `annual_margin_28_day_usd` = revenue - reagent_cost - shipper_cost
- `annual_margin_difference_28_minus_14_usd` = `annual_margin_28_day_usd - annual_margin_14_day_usd`

**CRITICAL**: The shipper cost formula is `shipper_cost_usd * runs_per_year`. It is NOT multiplied by active_labs. Re-read the task: the task says `annual_shipper_cost` without specifying per-lab multiplication. The previous run's massive error (26185 vs 1090) strongly suggests shipper cost was incorrectly multiplied by active_labs. However, inspect the test file first — if the test expects shipper cost multiplied by active_labs, do that instead. If the test file doesn't clarify, use `shipper_cost_usd * runs_per_year` only (no active_labs multiplier) as the default since the task instruction says `annual_shipper_cost` as a flat cost.

**ACTUALLY** — re-read the task more carefully. The task gives explicit formulas for annual_revenue and annual_reagent_cost (both include active_labs), but for annual_shipper_cost there is NO explicit formula given. The task just says `annual_margin = annual_revenue - annual_reagent_cost - annual_shipper_cost`. So we need to figure out what annual_shipper_cost is. The previous iteration produced 26185.96 instead of expected 1090.76. Look at the test file to see if it reveals expected values. If not, try both approaches and see which produces values consistent with the expected output.

To be safe: first check the test file for any expected numeric values. Use those to reverse-engineer the correct shipper cost formula. If the test reveals expected per-panel values, verify your formula against them.

### 2h. Compute totals
- `total_annual_margin_14_day_usd` = sum of all panels' `annual_margin_14_day_usd`
- `total_annual_margin_28_day_usd` = sum of all panels' `annual_margin_28_day_usd`
- `total_annual_margin_difference_28_minus_14_usd` = sum of all panels' differences
- `absolute_total_margin_difference_usd` = abs(total_annual_margin_difference_28_minus_14_usd)

### 2i. Decision
- If `absolute_total_margin_difference_usd < 6000`, decision = `"adopt_28_day"`
- Otherwise, decision = `"keep_14_day"`
- Write a justification string.

### 2j. Build output JSON
- Start with `report_template.json` as the base.
- Preserve `metadata` and `audit_notes` EXACTLY as they appear in the template (do not modify, reorder, or drop any fields).
- Fill in `analysis` with the schema specified. The `assumptions` block must contain EXACTLY these keys and values:
  - `runs_per_year_14_day`: 26
  - `runs_per_year_28_day`: 13
  - `switch_threshold_usd`: 6000
  - `override_rule`: `"highest numeric approved rev with non-empty active_labs, else default_active_labs"`
  - `holdout_rule`: `"exclude holdout_state=exclude"`
  - `adjustment_rule`: `"missing network_tier adjustment defaults to 0.0"`
- Do NOT add extra keys like `analysis_mode_filter` to assumptions.
- Each panel object in `analysis.panels` MUST include ALL keys from the schema, including `network_tier`.
- Sort panels by `panel_code` ascending.
- Round all USD values to 2 decimal places.

### 2k. Write output files
1. Write `/root/diagpanel_policy_report.json` with `json.dumps(report, indent=2)`.
2. Write `/root/diagpanel_policy_summary.md` with 4-8 non-empty lines including:
   - Total 14-day margin
   - Total 28-day margin
   - Absolute difference
   - Final decision slug (`adopt_28_day` or `keep_14_day`)
   - Format USD values as plain numbers with 2 decimals (e.g., `1234.56`), NOT with commas or dollar signs, unless the test expects a specific format. Check the test file for expected format patterns.

## Step 3: Run the script
```
cd /root && python solve.py
```

## Step 4: Validate
```
cat /root/diagpanel_policy_report.json
cat /root/diagpanel_policy_summary.md
python -c "import json; d=json.load(open('/root/diagpanel_policy_report.json')); print('Keys in assumptions:', list(d['analysis']['assumptions'].keys())); print('First panel keys:', list(d['analysis']['panels'][0].keys()) if d['analysis']['panels'] else 'NO PANELS')"
```

Check that:
- `assumptions` has exactly 6 keys (the ones listed above)
- Each panel has `network_tier` key
- No extra keys in assumptions

## Step 5: Run tests
```
cd /root && python -m pytest test_output.py -v 2>&1 | head -80
```

If tests fail, read the error messages carefully, fix the script, and re-run. Pay special attention to:
- Numeric value mismatches (recalculate manually if needed)
- Missing or extra JSON keys
- String format mismatches in the markdown summary

## Step 6: Debug numeric issues if needed

If numeric values don't match, add debug prints to solve.py showing intermediate calculations for each panel:
- base_payment, network_adjustment, active_labs, reagent_cost_per_1000, tests_per_run, shipper_cost
- Then the computed annual values

Compare with expected values from test output to identify which formula is wrong. Fix and re-run until tests pass.

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