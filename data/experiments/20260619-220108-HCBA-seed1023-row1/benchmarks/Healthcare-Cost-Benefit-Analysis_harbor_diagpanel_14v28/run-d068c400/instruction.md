# Task Instruction

Execute the following steps in order to produce `/root/diagpanel_policy_report.json` and `/root/diagpanel_policy_summary.md`.

## Step 1 — Inspect all input files

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

## Step 2 — Write and run a Python script

Create `/root/solve.py` with the following logic (use Python 3, `json`, `csv` standard libs):

### 2a. Load data
- Load `panel_manifest.json` → list of panel objects.
- Load `shipper_cost.csv` → dict keyed by `shipper_class` → `shipper_cost_usd` (float).
- Load `contract_terms.csv` → list of dicts.
- Load `network_adjustments.csv` → dict keyed by `network_tier` → `network_adjustment_per_run_per_lab_usd` (float).
- Load `lab_capacity_overrides.csv` → list of dicts.
- Load `holdouts.json` → list/dict of holdout entries.
- Load `report_template.json` → preserve `metadata` and `audit_notes` exactly.

### 2b. Filter panels
- Keep only panels where `analysis_mode` == `"review"`.
- Build a set of excluded panel_codes from holdouts where `holdout_state` == `"exclude"`.
- Remove any panel whose `panel_code` is in the excluded set.
- Call these the "retained panels".

### 2c. Resolve contract terms
For each retained panel:
- Collect contract rows where `status_flag` == `"current"` AND `panel_ref` matches either the panel's `panel_name` OR any entry in the panel's `alias_labels` list.
- If multiple rows match, keep the one with the latest `effective_week` (compare as strings if ISO-week format, or parse appropriately — inspect the data first).
- Extract `base_payment_per_run_per_lab_usd` (float) from the winning row.

### 2d. Resolve active labs
For each retained panel:
- From `lab_capacity_overrides.csv`, select rows where `panel_code` matches AND `approval` == `"approved"`.
- Among those, discard rows where `rev` is blank/empty or `active_labs` is blank/empty.
- Among remaining, keep the row with the highest numeric `rev` (convert to int or float for comparison).
- Use that row's `active_labs` (int).
- If no valid row exists, use `default_active_labs` from `panel_manifest.json` for that panel.

### 2e. Look up network adjustment
- For each panel's `network_tier`, look up `network_adjustment_per_run_per_lab_usd` from `network_adjustments.csv`.
- If the tier is not found, use `0.0`.

### 2f. Look up shipper cost
- For each panel's `shipper_class`, look up `shipper_cost_usd` from `shipper_cost.csv`.

### 2g. Compute per-panel financials
For each retained panel, compute:

```
total_payment_per_run_per_lab = base_payment + network_adjustment

# 14-day model
runs_14 = 26
tests_14 = tests_per_lab_per_run_14_day  (from manifest)
annual_revenue_14 = total_payment_per_run_per_lab * active_labs * runs_14
annual_reagent_cost_14 = reagent_cost_per_1000_tests * active_labs * tests_14 * runs_14 / 1000
annual_shipper_cost_14 = shipper_cost_usd * runs_14
annual_margin_14 = annual_revenue_14 - annual_reagent_cost_14 - annual_shipper_cost_14

# 28-day model
runs_28 = 13
tests_28 = tests_per_lab_per_run_28_day  (from manifest)
annual_revenue_28 = total_payment_per_run_per_lab * active_labs * runs_28
annual_reagent_cost_28 = reagent_cost_per_1000_tests * active_labs * tests_28 * runs_28 / 1000
annual_shipper_cost_28 = shipper_cost_usd * runs_28
annual_margin_28 = annual_revenue_28 - annual_reagent_cost_28 - annual_shipper_cost_28

difference = annual_margin_28 - annual_margin_14
```

**IMPORTANT**: `annual_shipper_cost = shipper_cost_usd * runs_per_year`. Shipper cost is per-shipment (per run), NOT per lab. This was a previous failure point.

### 2h. Totals and decision
```
total_margin_14 = sum of all annual_margin_14
total_margin_28 = sum of all annual_margin_28
total_difference = sum of all per-panel differences
absolute_difference = abs(total_difference)

if absolute_difference < 6000:
    decision = "adopt_28_day"
else:
    decision = "keep_14_day"
```

### 2i. Round all currency values to 2 decimal places.

### 2j. Build output JSON
- Sort panels by `panel_code` ascending.
- Use the exact schema from the task. Each panel object must have ALL listed keys.
- `metadata` and `audit_notes` come verbatim from `report_template.json`.
- `justification` should be a brief string citing the absolute difference vs threshold.
- Write to `/root/diagpanel_policy_report.json` with `json.dump(..., indent=2)`.

### 2k. Build summary markdown `/root/diagpanel_policy_summary.md`
- 4–8 non-empty lines.
- Must include total 14-day margin, total 28-day margin, absolute difference, and the exact decision slug.
- **Format all USD values with commas** (use `f"{value:,.2f}"` in Python). This was a previous failure point.
- Example format:
```
# Diagnostics Panel Policy Summary

Total 14-day annual margin: $XX,XXX.XX
Total 28-day annual margin: $XX,XXX.XX
Absolute margin difference: $X,XXX.XX
Decision: adopt_28_day
```

## Step 3 — Run the script
```bash
python3 /root/solve.py
```

## Step 4 — Validate outputs
```bash
python3 -c "
import json
with open('/root/diagpanel_policy_report.json') as f:
    d = json.load(f)
assert 'metadata' in d
assert 'audit_notes' in d
assert 'analysis' in d
assert 'assumptions' in d['analysis']
assert 'panels' in d['analysis']
assert 'totals' in d['analysis']
assert 'recommendation' in d['analysis']
assert 'decision' in d['analysis']['recommendation']
assert d['analysis']['recommendation']['decision'] in ('adopt_28_day', 'keep_14_day')
panels = d['analysis']['panels']
codes = [p['panel_code'] for p in panels]
assert codes == sorted(codes), 'Panels not sorted by panel_code'
for p in panels:
    for k in ['panel_code','panel_name','active_labs','reagent_cost_per_1000_tests_usd','network_tier','network_adjustment_per_run_per_lab_usd','shipper_class','shipper_cost_usd','base_payment_per_run_per_lab_usd','total_payment_per_run_per_lab_usd','tests_per_lab_per_run_14_day','tests_per_lab_per_run_28_day','annual_reagent_cost_14_day_usd','annual_reagent_cost_28_day_usd','annual_shipper_cost_14_day_usd','annual_shipper_cost_28_day_usd','annual_revenue_14_day_usd','annual_revenue_28_day_usd','annual_margin_14_day_usd','annual_margin_28_day_usd','annual_margin_difference_28_minus_14_usd']:
        assert k in p, f'Missing key {k} in panel {p.get("panel_code","?")}'  
print('JSON schema OK')
"

cat /root/diagpanel_policy_summary.md
wc -l /root/diagpanel_policy_summary.md
```

If any validation fails, inspect the error, fix `solve.py`, and re-run.

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