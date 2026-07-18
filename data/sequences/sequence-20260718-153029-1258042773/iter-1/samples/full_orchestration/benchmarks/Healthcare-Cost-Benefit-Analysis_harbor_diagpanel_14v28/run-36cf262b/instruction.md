# Task Instruction

## Task: Healthcare Cost-Benefit Analysis – 14-day vs 28-day Diagnostics Panel Policy

You must produce two output files:
1. `/root/diagpanel_policy_report.json`
2. `/root/diagpanel_policy_summary.md`

### Step 0 – Inspect all input files

Read and print the full contents of every input file before writing any code:
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
cat /root/tests/test_outputs.py 2>/dev/null || echo 'no test file found'
ls /root/tests/ 2>/dev/null || echo 'no tests dir'
```

Study the data carefully before proceeding. Note exact field names, data types, and edge cases.

### Step 1 – Write a Python script `/root/solve.py`

Write a single Python script that does all of the following:

#### 1a. Load data
- Load `panel_manifest.json`, `shipper_cost.csv`, `contract_terms.csv`, `network_adjustments.csv`, `lab_capacity_overrides.csv`, `holdouts.json`, `report_template.json`.

#### 1b. Determine retained panels
- From `panel_manifest.json`, keep only panels where `analysis_mode == "review"`.
- From `holdouts.json`, exclude any panel whose `holdout_state == "exclude"`. Match by `panel_code`.

#### 1c. Resolve contract terms
- For each retained panel, find rows in `contract_terms.csv` where `panel_ref` matches either `panel_name` or any entry in `alias_labels` (from the manifest).
- Keep only rows with `status_flag == "current"`.
- If multiple current rows match the same panel, keep the one with the latest `effective_week` (parse as date or string – compare lexicographically if ISO format).
- Extract `base_payment_per_run_per_lab_usd` from the winning row.

#### 1d. Network adjustment
- Match the panel's `network_tier` to `network_adjustments.csv` to get `network_adjustment_per_run_per_lab_usd`.
- If the tier is not found, use `0.0`.

#### 1e. Active labs
- From `lab_capacity_overrides.csv`, keep rows where `approval == "approved"`, and where `rev` is not blank/empty and `active_labs` is not blank/empty.
- Convert `rev` to numeric. If multiple valid rows exist for the same `panel_code`, keep the one with the highest `rev`.
- If no valid override row exists for a panel, use `default_active_labs` from the manifest.

#### 1f. Shipper cost
- Match the panel's `shipper_class` to `shipper_cost.csv` to get `shipper_cost_usd`.

#### 1g. Compute per-panel metrics (round all USD to 2 decimals at the end)
- `total_payment_per_run_per_lab_usd = base_payment_per_run_per_lab_usd + network_adjustment_per_run_per_lab_usd`
- 14-day model: 26 runs/year, `tests_per_lab_per_run_14_day`
- 28-day model: 13 runs/year, `tests_per_lab_per_run_28_day`
- `annual_revenue = total_payment_per_run_per_lab_usd * active_labs * runs_per_year`
- `annual_reagent_cost = reagent_cost_per_1000_tests_usd * active_labs * tests_per_lab_per_run * runs_per_year / 1000`
- `annual_shipper_cost_14_day = shipper_cost_usd * 26` (shipper cost is per shipment, one per run)
- `annual_shipper_cost_28_day = shipper_cost_usd * 13`
  **IMPORTANT**: Re-read the task. The shipper cost formula is not explicitly given per-panel beyond `shipper_cost_usd` from the CSV. Look at the report schema: there is `annual_shipper_cost_14_day_usd` and `annual_shipper_cost_28_day_usd`. The annual margin formula says `annual_revenue - annual_reagent_cost - annual_shipper_cost`. Since shipper cost is per shipment and there are `runs_per_year` shipments, compute: `annual_shipper_cost = shipper_cost_usd * runs_per_year`. (This is the most logical interpretation – shipper cost per run times number of runs.)
- `annual_margin = annual_revenue - annual_reagent_cost - annual_shipper_cost`
- `annual_margin_difference_28_minus_14 = annual_margin_28_day - annual_margin_14_day`

Round all currency values to 2 decimal places.

#### 1h. Totals
- Sum all per-panel `annual_margin_14_day_usd` → `total_annual_margin_14_day_usd`
- Sum all per-panel `annual_margin_28_day_usd` → `total_annual_margin_28_day_usd`
- `total_annual_margin_difference_28_minus_14_usd = total_28 - total_14`
- `absolute_total_margin_difference_usd = abs(total_difference)`
- Round all to 2 decimals.

#### 1i. Decision
- If `absolute_total_margin_difference_usd < 6000` → `adopt_28_day`
- Otherwise → `keep_14_day`
- Justification: a short string explaining the decision referencing the threshold and the absolute difference.

#### 1j. Build JSON output
- **CRITICAL**: The `recommendation` field must be a nested object: `{"decision": "...", "justification": "..."}` inside `analysis.recommendation`. Do NOT put `decision` and `justification` at the `analysis` level.
- Copy `metadata` and `audit_notes` from `report_template.json` exactly as-is.
- `assumptions` must match the schema exactly with these fixed values.
- `panels` list sorted by `panel_code` ascending.
- Write to `/root/diagpanel_policy_report.json` with `json.dump(..., indent=2)`.

#### 1k. Build markdown summary
- Write `/root/diagpanel_policy_summary.md` with 4-8 non-empty lines.
- Must include: total 14-day margin (USD), total 28-day margin (USD), absolute difference (USD), and the exact decision slug (`adopt_28_day` or `keep_14_day`).

### Step 2 – Run the script
```bash
cd /root && python solve.py
```

### Step 3 – Validate outputs
- `cat /root/diagpanel_policy_report.json` – verify JSON is valid, `recommendation` is nested correctly, `panels` are sorted by `panel_code`, all currency values have 2 decimal places, `metadata` and `audit_notes` match the template.
- `cat /root/diagpanel_policy_summary.md` – verify 4-8 non-empty lines with required content.
- If tests exist: `cd /root && python -m pytest tests/ -v`

### Step 4 – Fix any issues
If tests fail or output looks wrong, re-read the error, fix the script, re-run, and re-validate. Pay special attention to:
- Schema structure (especially `recommendation` nesting)
- Contract term matching (check `alias_labels` carefully – it may be a list)
- Lab capacity override logic (blank `rev` or `active_labs` filtering)
- Shipper cost interpretation
- Rounding

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