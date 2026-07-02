# Task Instruction

## Task: Healthcare Cost-Benefit Analysis — diagpanel_14v28

You must produce two output files by reading and processing several input files. Follow every step carefully.

### Step 1: Read all input files

Read these files and print/inspect their contents:
- `/root/panel_manifest.json`
- `/root/shipper_cost.csv`
- `/root/contract_terms.csv`
- `/root/network_adjustments.csv`
- `/root/lab_capacity_overrides.csv`
- `/root/holdouts.json`
- `/root/report_template.json`

### Step 2: Write a Python script `/root/solve.py` that does the following

#### 2a. Load data
- Load `panel_manifest.json` — an array of panel objects.
- Load `holdouts.json` — an array of holdout objects.
- Load `report_template.json` — a JSON object with `metadata` and `audit_notes`.
- Load `shipper_cost.csv`, `contract_terms.csv`, `network_adjustments.csv`, `lab_capacity_overrides.csv` as CSV.

#### 2b. Determine retained panels
- From `panel_manifest.json`, keep only panels where `analysis_mode == "review"`.
- From those, exclude any panel whose `panel_code` appears in `holdouts.json` with `holdout_state == "exclude"`.
- The remaining panels are "retained panels".

#### 2c. Resolve contract terms for each retained panel
- For each row in `contract_terms.csv`, check if `panel_ref` matches either the panel's `panel_name` OR any entry in the panel's `alias_labels` list.
- Use only rows where `status_flag == "current"`.
- If multiple current rows match, keep the one with the latest `effective_week` (compare as strings if ISO date-like, or parse appropriately).
- Extract `base_payment_per_run_per_lab_usd` from the matched contract row.

#### 2d. Resolve network adjustment
- Each retained panel has a `network_tier` field.
- Look up `network_adjustment_per_run_per_lab_usd` from `network_adjustments.csv` by matching `network_tier`.
- If the tier is not found, use `0.0`.

#### 2e. Resolve shipper cost
- Each retained panel has a `shipper_class` field.
- Look up `shipper_cost_usd` from `shipper_cost.csv` by matching `shipper_class`.

#### 2f. Resolve active labs
- From `lab_capacity_overrides.csv`, keep only rows where `approval == "approved"`.
- Among those, ignore rows where `rev` is blank/empty or `active_labs` is blank/empty.
- If multiple valid rows exist for the same `panel_code`, keep the one with the highest numeric `rev`.
- If a retained panel has no valid override row, use `default_active_labs` from `panel_manifest.json`.

#### 2g. Compute per-panel financials
For each retained panel:
- `total_payment_per_run_per_lab_usd = base_payment_per_run_per_lab_usd + network_adjustment_per_run_per_lab_usd`
- 14-day model: `runs_per_year = 26`, use `tests_per_lab_per_run_14_day`
- 28-day model: `runs_per_year = 13`, use `tests_per_lab_per_run_28_day`
- `annual_revenue = total_payment_per_run_per_lab_usd * active_labs * runs_per_year`
- `annual_reagent_cost = reagent_cost_per_1000_tests_usd * active_labs * tests_per_lab_per_run * runs_per_year / 1000`
- `annual_shipper_cost = shipper_cost_usd * runs_per_year`  (NOTE: shipper cost is per shipment/run, so `shipper_cost_usd * runs_per_year`. But re-check the formula — the instructions say `annual_margin = annual_revenue - annual_reagent_cost - annual_shipper_cost`. The shipper cost from the CSV is a flat per-shipment cost. Compute `annual_shipper_cost_14_day = shipper_cost_usd * 26` and `annual_shipper_cost_28_day = shipper_cost_usd * 13`.)
- `annual_margin = annual_revenue - annual_reagent_cost - annual_shipper_cost`
- `annual_margin_difference_28_minus_14 = annual_margin_28_day - annual_margin_14_day`
- Round ALL currency values to 2 decimal places.

#### 2h. Compute totals
- `total_annual_margin_14_day_usd` = sum of all panels' `annual_margin_14_day_usd`
- `total_annual_margin_28_day_usd` = sum of all panels' `annual_margin_28_day_usd`
- `total_annual_margin_difference_28_minus_14_usd` = sum of all panels' `annual_margin_difference_28_minus_14_usd`
- `absolute_total_margin_difference_usd` = `abs(total_annual_margin_difference_28_minus_14_usd)`
- Round all to 2 decimals.

#### 2i. Decision
- If `absolute_total_margin_difference_usd < 6000`, decision = `adopt_28_day`
- Otherwise, decision = `keep_14_day`
- Justification: a short string explaining the decision referencing the threshold and the absolute difference.

#### 2j. Build output JSON
- Start with `metadata` and `audit_notes` copied EXACTLY from `report_template.json`.
- Build `analysis` with `assumptions`, `panels` (sorted by `panel_code` ascending), `totals`, and `recommendation`.
- The `assumptions` object must have exactly:
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
- Each panel object in `panels` must have ALL fields from the schema (panel_code, panel_name, active_labs, reagent_cost_per_1000_tests_usd, network_tier, network_adjustment_per_run_per_lab_usd, shipper_class, shipper_cost_usd, base_payment_per_run_per_lab_usd, total_payment_per_run_per_lab_usd, tests_per_lab_per_run_14_day, tests_per_lab_per_run_28_day, annual_reagent_cost_14_day_usd, annual_reagent_cost_28_day_usd, annual_shipper_cost_14_day_usd, annual_shipper_cost_28_day_usd, annual_revenue_14_day_usd, annual_revenue_28_day_usd, annual_margin_14_day_usd, annual_margin_28_day_usd, annual_margin_difference_28_minus_14_usd).
- Write to `/root/diagpanel_policy_report.json` with `json.dump(..., indent=2)`.

#### 2k. Build summary markdown
Write `/root/diagpanel_policy_summary.md` with 4-8 non-empty lines including:
- Total 14-day margin (USD)
- Total 28-day margin (USD)
- Absolute difference (USD)
- Final decision using the exact slug `adopt_28_day` or `keep_14_day`

### Step 3: Run the script
```bash
cd /root && python solve.py
```

### Step 4: Validate outputs
- `cat /root/diagpanel_policy_report.json` and verify:
  - `metadata` and `audit_notes` match the template exactly.
  - `panels` is sorted by `panel_code`.
  - All currency values are rounded to 2 decimals.
  - The decision logic is correct.
- `cat /root/diagpanel_policy_summary.md` and verify it has 4-8 non-empty lines with the required info.

### IMPORTANT NOTES
- Before writing the script, carefully inspect ALL input files to understand their exact field names, data types, and structure.
- Pay special attention to how `panel_ref` in `contract_terms.csv` maps to panels (could match `panel_name` or any element of `alias_labels`).
- The `active_labs` field should be an integer in the output.
- `tests_per_lab_per_run_14_day` and `tests_per_lab_per_run_28_day` should be integers.
- `shipper_cost_usd` from CSV is the cost per shipment. Annual shipper cost = shipper_cost_usd × runs_per_year.
- If anything is ambiguous in the shipper cost formula, note that the task says `annual_margin = annual_revenue - annual_reagent_cost - annual_shipper_cost` and the only shipper cost input is `shipper_cost_usd` from `shipper_cost.csv`. This is a per-run cost, so multiply by runs_per_year to get annual.
- Double-check: the `annual_shipper_cost` fields appear in per-panel output. The shipper cost is matched by `shipper_class` from the panel manifest to `shipper_cost.csv`. It is a per-run (per-shipment) cost.

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