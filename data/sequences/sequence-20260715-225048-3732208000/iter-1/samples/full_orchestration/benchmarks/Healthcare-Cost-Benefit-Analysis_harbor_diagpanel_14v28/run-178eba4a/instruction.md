# Task Instruction

You are tasked with producing a healthcare cost-benefit analysis comparing 14-day vs 28-day replenishment cadences for diagnostic panels.

## Step 1: Inspect all input files

Read and display the contents of every input file:
```
cat /root/panel_manifest.json
cat /root/shipper_cost.csv
cat /root/contract_terms.csv
cat /root/network_adjustments.csv
cat /root/lab_capacity_overrides.csv
cat /root/holdouts.json
cat /root/report_template.json
```

## Step 2: Write and execute a Python script

Write a single Python script `/root/solve.py` that does all of the following:

### 2a. Load data
- Load `panel_manifest.json`, `holdouts.json`, `report_template.json` as JSON.
- Load `shipper_cost.csv`, `contract_terms.csv`, `network_adjustments.csv`, `lab_capacity_overrides.csv` as CSV (use pandas or csv module).

### 2b. Filter panels
- From `panel_manifest.json`, keep only panels where `analysis_mode == "review"`.
- From `holdouts.json`, exclude any panel whose `panel_code` has `holdout_state == "exclude"`.
- The remaining panels are the "retained" panels.

### 2c. Resolve contract terms
- For each retained panel, find matching rows in `contract_terms.csv` by checking if the CSV's `panel_ref` matches either the panel's `panel_name` OR any entry in the panel's `alias_labels` list.
- Keep only rows where `status_flag == "current"`.
- If multiple current rows match the same panel, keep the one with the latest `effective_week` (compare as strings if they look like date-like week identifiers, or parse appropriately).
- Extract `base_payment_per_run_per_lab_usd` from the winning contract row.

### 2d. Resolve network adjustment
- Match the panel's `network_tier` to `network_adjustments.csv` to get `network_adjustment_per_run_per_lab_usd`.
- If the tier is not found, use `0.0`.

### 2e. Resolve active labs
- From `lab_capacity_overrides.csv`, keep only rows where `approval == "approved"`.
- Discard rows where `rev` is blank/empty or `active_labs` is blank/empty.
- Convert `rev` to numeric. If multiple valid rows exist for the same `panel_code`, keep the one with the highest `rev`.
- If a retained panel has no valid override row, use `default_active_labs` from `panel_manifest.json`.

### 2f. Resolve shipper cost
- Match the panel's `shipper_class` to `shipper_cost.csv` to get `shipper_cost_usd`.

### 2g. Compute per-panel financials
For each retained panel:
- `total_payment_per_run_per_lab_usd = base_payment_per_run_per_lab_usd + network_adjustment_per_run_per_lab_usd`
- 14-day model: `runs_per_year = 26`, use `tests_per_lab_per_run_14_day`
- 28-day model: `runs_per_year = 13`, use `tests_per_lab_per_run_28_day`
- `annual_revenue = total_payment_per_run_per_lab_usd * active_labs * runs_per_year`
- `annual_reagent_cost = reagent_cost_per_1000_tests_usd * active_labs * tests_per_lab_per_run * runs_per_year / 1000`
- `annual_shipper_cost = shipper_cost_usd` (this is the annual shipper cost for the given cadence — **IMPORTANT**: check if shipper_cost_usd is per-shipment or annual. Read the CSV carefully. If it's per-shipment, multiply by runs_per_year. If it looks annual, use as-is. The formula says `annual_margin = annual_revenue - annual_reagent_cost - annual_shipper_cost`, so shipper cost must be annual. If the CSV value appears to be per-shipment, multiply by runs_per_year to get annual. If unsure, check the magnitude relative to other costs.)
- `annual_margin = annual_revenue - annual_reagent_cost - annual_shipper_cost`
- `annual_margin_difference_28_minus_14 = annual_margin_28_day - annual_margin_14_day`

### 2h. Compute totals and decision
- Sum all per-panel 14-day margins → `total_annual_margin_14_day_usd`
- Sum all per-panel 28-day margins → `total_annual_margin_28_day_usd`
- `total_annual_margin_difference_28_minus_14_usd = total_28 - total_14`
- `absolute_total_margin_difference_usd = abs(total_difference)`
- If `absolute_total_margin_difference_usd < 6000` → decision = `adopt_28_day`
- Otherwise → decision = `keep_14_day`

### 2i. Build JSON output
- Use the exact schema from the task instructions.
- Copy `metadata` and `audit_notes` from `report_template.json` exactly as-is.
- Populate `analysis.assumptions` with the exact keys and values shown in the schema.
- Sort `analysis.panels` by `panel_code` ascending (alphabetical).
- Round ALL currency values to 2 decimal places.
- `active_labs` should be an integer.
- `tests_per_lab_per_run_14_day` and `tests_per_lab_per_run_28_day` should be integers (or match their source type).
- Write to `/root/diagpanel_policy_report.json` with `json.dump(..., indent=2)`.

### 2j. Build Markdown summary
- Write `/root/diagpanel_policy_summary.md` with 4-8 non-empty lines.
- Must include: total 14-day margin (USD), total 28-day margin (USD), absolute difference (USD), and the exact decision slug (`adopt_28_day` or `keep_14_day`).
- Format currency values with 2 decimal places.

## Step 3: Execute the script
```
cd /root && python solve.py
```

## Step 4: Validate outputs
```
cat /root/diagpanel_policy_report.json
cat /root/diagpanel_policy_summary.md
python -c "import json; d=json.load(open('/root/diagpanel_policy_report.json')); print('panels:', len(d['analysis']['panels'])); print('decision:', d['analysis']['recommendation']['decision']); print('totals:', d['analysis']['totals']); print('metadata:', d['metadata']); print('audit_notes:', d['audit_notes'])"
```

Verify:
- JSON is valid and parseable
- `metadata` and `audit_notes` match `report_template.json` exactly
- `assumptions` has all required keys including `switch_threshold_usd: 6000`
- Panels are sorted by `panel_code`
- All currency values have exactly 2 decimal places
- The summary has 4-8 non-empty lines with required content
- Decision slug is exactly `adopt_28_day` or `keep_14_day`

If anything looks wrong, fix and re-run. Pay special attention to:
- Alias matching (check `alias_labels` carefully)
- The shipper cost interpretation (per-shipment vs annual)
- The `effective_week` comparison for picking latest contract
- Numeric `rev` comparison for lab capacity overrides
- Edge cases: missing network tier, missing override rows

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