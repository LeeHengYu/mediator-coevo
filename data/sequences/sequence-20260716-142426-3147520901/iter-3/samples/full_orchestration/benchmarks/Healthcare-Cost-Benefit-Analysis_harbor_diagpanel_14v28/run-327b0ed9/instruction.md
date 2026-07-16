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

Create `/root/solve.py` with the logic below, then run it with `python3 /root/solve.py`.

The script must:

### 2a – Load data
- Load `panel_manifest.json` (list of panel objects).
- Load `shipper_cost.csv`, `contract_terms.csv`, `network_adjustments.csv`, `lab_capacity_overrides.csv` as CSV (use `csv.DictReader`).
- Load `holdouts.json` (list of holdout objects).
- Load `report_template.json`.

### 2b – Filter panels
- Keep only panels where `analysis_mode == "review"`.
- Build a set of excluded panel_codes from `holdouts.json` where `holdout_state == "exclude"`.
- Remove any panel whose `panel_code` is in the excluded set.
- Call the remaining panels "retained panels".

### 2c – Resolve contract terms
For each retained panel:
- Build a lookup set containing the panel's `panel_name` plus every entry in its `alias_labels` list (if present).
- Find all rows in `contract_terms.csv` where `panel_ref` matches any value in that lookup set AND `status_flag == "current"`.
- If multiple matching rows, keep the one with the latest `effective_week` (compare as strings; they should be ISO-like dates or YYYY-WNN — just use lexicographic max).
- Extract `base_payment_per_run_per_lab_usd` (convert to float).

### 2d – Network adjustment
- Build a dict from `network_adjustments.csv`: key = `network_tier`, value = float(`network_adjustment_per_run_per_lab_usd`).
- For each retained panel, look up its `network_tier`. If not found, use 0.0.

### 2e – Active labs
- Parse `lab_capacity_overrides.csv`. Keep only rows where `approval == "approved"` AND `rev` is not blank/empty AND `active_labs` is not blank/empty.
- Convert `rev` to int/float for numeric comparison.
- Group by `panel_code`; keep the row with the highest numeric `rev`.
- For each retained panel, if its `panel_code` appears in the override dict, use that `active_labs` (int). Otherwise use `default_active_labs` from `panel_manifest.json` (int).

### 2f – Shipper cost
- Build a dict from `shipper_cost.csv`: key = `shipper_class`, value = float(`shipper_cost_usd`).
- For each retained panel, look up `shipper_class` to get `shipper_cost_usd`.

### 2g – Calculations (per panel)
For each retained panel, using:
- `runs_14 = 26`, `runs_28 = 13`
- `tests_14 = tests_per_lab_per_run_14_day` (from manifest, int)
- `tests_28 = tests_per_lab_per_run_28_day` (from manifest, int)
- `reagent = reagent_cost_per_1000_tests_usd` (from manifest, float)
- `labs = active_labs` (int)
- `base = base_payment_per_run_per_lab_usd` (float)
- `adj = network_adjustment_per_run_per_lab_usd` (float)
- `total_payment = base + adj`
- `shipper = shipper_cost_usd` (float)

Compute:
```
annual_revenue_14 = total_payment * labs * 26
annual_revenue_28 = total_payment * labs * 13
annual_reagent_cost_14 = reagent * labs * tests_14 * 26 / 1000
annual_reagent_cost_28 = reagent * labs * tests_28 * 13 / 1000
annual_shipper_cost_14 = shipper * 26
annual_shipper_cost_28 = shipper * 13
annual_margin_14 = annual_revenue_14 - annual_reagent_cost_14 - annual_shipper_cost_14
annual_margin_28 = annual_revenue_28 - annual_reagent_cost_28 - annual_shipper_cost_28
difference = annual_margin_28 - annual_margin_14
```
Round every currency value to 2 decimal places.

### 2h – Totals and decision
- Sum all per-panel `annual_margin_14` → `total_annual_margin_14_day_usd`
- Sum all per-panel `annual_margin_28` → `total_annual_margin_28_day_usd`
- `total_diff = total_28 - total_14`
- `abs_diff = abs(total_diff)`
- Round each to 2 decimals.
- If `abs_diff < 6000`: decision = `adopt_28_day`, else `keep_14_day`.
- Justification: a short sentence stating the absolute difference vs the $6,000 threshold.

### 2i – Build JSON output
- Start from `report_template.json`. Preserve `metadata` and `audit_notes` exactly as they appear.
- Set `analysis.assumptions` to the exact dict:
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
- Build `analysis.panels` list sorted by `panel_code` ascending. Each entry has every field from the schema with correct types (strings for codes/names/tiers, ints for labs and tests, floats rounded to 2 for all USD values).
- Set `analysis.totals` and `analysis.recommendation`.
- Write to `/root/diagpanel_policy_report.json` with `json.dump(..., indent=2)`.

### 2j – Build Markdown summary
Write `/root/diagpanel_policy_summary.md` with 4-8 non-empty lines including:
- Total 14-day margin in USD (e.g., `Total 14-day annual margin: $XX,XXX.XX`)
- Total 28-day margin in USD
- Absolute difference in USD
- Final decision using the exact slug `adopt_28_day` or `keep_14_day`

Use f-string formatting with `{value:,.2f}` — values must be floats, not strings.

## Step 3 – Validate outputs

After the script runs:
```
python3 -c "
import json
with open('/root/diagpanel_policy_report.json') as f:
    r = json.load(f)
assert 'metadata' in r and 'audit_notes' in r and 'analysis' in r
a = r['analysis']
assert 'assumptions' in a and 'panels' in a and 'totals' in a and 'recommendation' in a
assert a['assumptions']['switch_threshold_usd'] == 6000
assert isinstance(a['panels'], list) and len(a['panels']) > 0
codes = [p['panel_code'] for p in a['panels']]
assert codes == sorted(codes), 'panels not sorted by panel_code'
for p in a['panels']:
    assert isinstance(p['active_labs'], int)
    assert isinstance(p['annual_revenue_14_day_usd'], float)
print('JSON schema spot-check passed')
print('Decision:', a['recommendation']['decision'])
print('Panels:', len(a['panels']))
print('Total diff:', a['totals']['total_annual_margin_difference_28_minus_14_usd'])
"
```

```
cat /root/diagpanel_policy_summary.md
```

Verify the summary has 4-8 non-empty lines and contains the decision slug and all three required dollar figures.

If anything fails, debug by re-reading the input files and fix the script before re-running.

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