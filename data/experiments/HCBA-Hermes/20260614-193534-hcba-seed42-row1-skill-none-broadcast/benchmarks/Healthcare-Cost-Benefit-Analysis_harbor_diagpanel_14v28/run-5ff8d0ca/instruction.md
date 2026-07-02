# Task Instruction

Execute the following steps carefully to produce `/root/diagpanel_policy_report.json` and `/root/diagpanel_policy_summary.md`.

## Step 0 — Inspect all input files

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
Also inspect the test file to understand verifier expectations:
```
find /root -name '*.py' | head -20
cat /root/tests/test_outputs.py 2>/dev/null || cat /tests/test_outputs.py 2>/dev/null || find / -path '*/test*' -name '*.py' 2>/dev/null | head -10
```

## Step 1 — Write a Python script

Create `/root/solve.py` that does everything below. Run it with `python3 /root/solve.py`.

### 1a. Load data
- Load `panel_manifest.json` (list of panel objects).
- Load `shipper_cost.csv`, `contract_terms.csv`, `network_adjustments.csv`, `lab_capacity_overrides.csv` as CSV (use csv.DictReader).
- Load `holdouts.json` (list of holdout objects).
- Load `report_template.json`.

### 1b. Filter panels to review set
- Keep only panels where `analysis_mode` == `"review"`.
- Build a set of excluded panel_codes from `holdouts.json` where `holdout_state` == `"exclude"`.
- Remove any panel whose `panel_code` is in that excluded set.
- These are the "retained" panels.

### 1c. Match contract terms
For each retained panel:
- The panel has `panel_name` and `alias_labels` (a list of strings).
- A contract row matches if its `panel_ref` equals the panel's `panel_name` OR any entry in `alias_labels`.
- Keep only contract rows where `status_flag` == `"current"`.
- If multiple current rows match, keep the one with the latest `effective_week` (compare as strings — they should be ISO-like date strings; if purely numeric, compare numerically; use string comparison which works for ISO dates).
- Extract `base_payment_per_run_per_lab_usd` (float) from the winning contract row.

### 1d. Network adjustment
- Build a dict from `network_adjustments.csv`: key = `network_tier`, value = float(`network_adjustment_per_run_per_lab_usd`).
- For each retained panel, look up its `network_tier` from the manifest. If the tier is not in the dict, use 0.0.

### 1e. Active labs (lab_capacity_overrides)
- Filter to rows where `approval` == `"approved"`.
- Discard rows where `rev` is blank/empty or `active_labs` is blank/empty.
- Group remaining rows by `panel_code`.
- For each panel_code, keep the row with the highest numeric `rev` (convert to int or float).
- For each retained panel: if it has an override row, use that `active_labs` (int). Otherwise use `default_active_labs` from the manifest (int).

### 1f. Shipper cost
- Build a dict from `shipper_cost.csv`: key = `shipper_class`, value = float(`shipper_cost_usd`).
- Each panel has a `shipper_class` in the manifest. Look up the cost.

### 1g. Compute per-panel figures
For each retained panel, compute:
- `total_payment_per_run_per_lab_usd` = base_payment + network_adjustment
- `tests_per_lab_per_run_14_day` from manifest (int)
- `tests_per_lab_per_run_28_day` from manifest (int)
- `reagent_cost_per_1000_tests_usd` from manifest (float)
- 14-day model (runs_per_year=26):
  - `annual_revenue_14_day_usd` = total_payment * active_labs * 26
  - `annual_reagent_cost_14_day_usd` = reagent_cost_per_1000 * active_labs * tests_per_lab_per_run_14_day * 26 / 1000
  - `annual_shipper_cost_14_day_usd` = shipper_cost_usd * 26
  - `annual_margin_14_day_usd` = revenue - reagent_cost - shipper_cost
- 28-day model (runs_per_year=13): same formulas with 13 and tests_per_lab_per_run_28_day
- `annual_margin_difference_28_minus_14_usd` = margin_28 - margin_14

Note: shipper cost is per shipment (per run), so annual_shipper_cost = shipper_cost_usd * runs_per_year. (Verify this interpretation against the data — if the shipper_cost seems to already be annual, adjust. But the standard interpretation for per-run shipping is shipper_cost * runs_per_year.)

### 1h. Totals and decision
- `total_annual_margin_14_day_usd` = sum of all panels' margin_14
- `total_annual_margin_28_day_usd` = sum of all panels' margin_28
- `total_annual_margin_difference_28_minus_14_usd` = sum of all per-panel differences
- `absolute_total_margin_difference_usd` = abs(total_difference)
- Decision: if abs(total_difference) < 6000 → `"adopt_28_day"`, else `"keep_14_day"`

### 1i. Build JSON output
- Start from report_template.json. Preserve `metadata` and `audit_notes` exactly.
- Build `analysis` with:
  - `assumptions` dict with exactly these keys and values:
    - `runs_per_year_14_day`: 26
    - `runs_per_year_28_day`: 13
    - `switch_threshold_usd`: 6000
    - `override_rule`: `"highest numeric approved rev with non-empty active_labs, else default_active_labs"`
    - `holdout_rule`: `"exclude holdout_state=exclude"`
    - `adjustment_rule`: `"missing network_tier adjustment defaults to 0.0"`
  - `panels` list sorted by `panel_code` ascending. Each panel object must include ALL of these keys:
    - `panel_code`, `panel_name`, `active_labs`, `reagent_cost_per_1000_tests_usd`, `network_tier`, `network_adjustment_per_run_per_lab_usd`, `shipper_class`, `shipper_cost_usd`, `base_payment_per_run_per_lab_usd`, `total_payment_per_run_per_lab_usd`, `tests_per_lab_per_run_14_day`, `tests_per_lab_per_run_28_day`, `annual_reagent_cost_14_day_usd`, `annual_reagent_cost_28_day_usd`, `annual_shipper_cost_14_day_usd`, `annual_shipper_cost_28_day_usd`, `annual_revenue_14_day_usd`, `annual_revenue_28_day_usd`, `annual_margin_14_day_usd`, `annual_margin_28_day_usd`, `annual_margin_difference_28_minus_14_usd`
  - All currency values rounded to 2 decimals.
  - `totals` dict with the four total keys, all rounded to 2 decimals.
  - `recommendation` with `decision` and `justification` (a short string explaining the decision).
- Write to `/root/diagpanel_policy_report.json` with `json.dump(..., indent=2)`.

### 1j. Build Markdown summary
Write `/root/diagpanel_policy_summary.md` with 4–8 non-empty lines. Use comma-separated thousands formatting for all currency values (e.g., `f'{value:,.2f}'`). Must include:
- Total 14-day margin line
- Total 28-day margin line  
- Absolute difference line
- Decision line using the exact slug `adopt_28_day` or `keep_14_day`

Example format:
```
# Diagnostics Panel Policy Summary

Total 14-day annual margin: $X,XXX.XX USD
Total 28-day annual margin: $X,XXX.XX USD
Absolute margin difference: $X,XXX.XX USD
Recommendation: adopt_28_day
```

## Step 2 — Run the script
```
cd /root && python3 solve.py
```

## Step 3 — Validate outputs
- `cat /root/diagpanel_policy_report.json` — check all panel keys are present, panels sorted by panel_code, metadata/audit_notes preserved.
- `cat /root/diagpanel_policy_summary.md` — check comma formatting and all required content.
- Run the test suite:
```
cd / && python3 -m pytest tests/test_outputs.py -v 2>/dev/null || cd /root && python3 -m pytest tests/test_outputs.py -v 2>/dev/null || find / -name 'test_output*' -exec python3 -m pytest {} -v \;
```

## Step 4 — Fix any failures
If any test fails, read the error carefully, re-read the relevant input files, fix the logic in solve.py, re-run, and re-validate. Pay special attention to:
- Missing keys in panel objects (previous failure #1)
- Contract matching via alias_labels (previous failure #2 — the total was way off, suggesting wrong panels or wrong contract rows were matched)
- Comma formatting in markdown (previous failure #3)
- The `effective_week` comparison — print the matched contract rows to debug
- The shipper cost formula — check whether it should be multiplied by runs_per_year or is already annual

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