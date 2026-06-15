# Task Instruction

Execute the following steps in order:

## Step 1: Inspect all input files

Read and display the contents of each input file:
```
cat /root/panel_manifest.json
cat /root/shipper_cost.csv
cat /root/contract_terms.csv
cat /root/network_adjustments.csv
cat /root/lab_capacity_overrides.csv
cat /root/holdouts.json
cat /root/report_template.json
```

## Step 2: Write and run a Python script

Create `/root/solve.py` that does the following:

1. **Load all input files** using `json` and `csv` modules.

2. **Filter panels**: From `panel_manifest.json`, keep only panels where `analysis_mode == "review"`.

3. **Apply holdouts**: Remove any panel whose `panel_code` appears in `holdouts.json` with `holdout_state == "exclude"`.

4. **Resolve contract terms**: For each retained panel, find matching rows in `contract_terms.csv` where `panel_ref` matches either the panel's `panel_name` OR any entry in the panel's `alias_labels` list. Among matches, keep only rows where `status_flag == "current"`. If multiple current rows match the same panel, keep the one with the latest `effective_week` (compare as strings if ISO date format, or parse appropriately). Extract `base_payment_per_run_per_lab_usd` from the winning row.

5. **Network adjustments**: From `network_adjustments.csv`, build a lookup by `network_tier`. For each retained panel, look up `network_adjustment_per_run_per_lab_usd` by the panel's `network_tier`. If not found, use `0.0`.

6. **Active labs from overrides**: From `lab_capacity_overrides.csv`, for each panel_code:
   - Keep only rows where `approval == "approved"`.
   - Among those, discard rows where `rev` is blank/empty or `active_labs` is blank/empty.
   - If multiple valid rows remain for the same `panel_code`, keep the one with the highest numeric `rev`.
   - Use that row's `active_labs` (as integer).
   - If no valid override row exists, use `default_active_labs` from `panel_manifest.json`.

7. **Shipper cost**: From `shipper_cost.csv`, build a lookup by `shipper_class`. For each panel, look up `shipper_cost_usd` by the panel's `shipper_class`.

8. **Compute per-panel metrics** (all currency values rounded to 2 decimals):
   - `total_payment_per_run_per_lab_usd = base_payment + network_adjustment`
   - 14-day model: `runs_per_year = 26`, use `tests_per_lab_per_run_14_day`
   - 28-day model: `runs_per_year = 13`, use `tests_per_lab_per_run_28_day`
   - `annual_revenue = total_payment_per_run_per_lab_usd * active_labs * runs_per_year`
   - `annual_reagent_cost = reagent_cost_per_1000_tests_usd * active_labs * tests_per_lab_per_run * runs_per_year / 1000`
   - `annual_shipper_cost = shipper_cost_usd * runs_per_year` (note: shipper cost is per shipment, so it's shipper_cost_usd * runs_per_year — but re-check the formula: the instructions say annual_margin = annual_revenue - annual_reagent_cost - annual_shipper_cost. The shipper_cost_usd from the CSV is likely per-shipment. Compute `annual_shipper_cost = shipper_cost_usd * runs_per_year`.)
   - `annual_margin = annual_revenue - annual_reagent_cost - annual_shipper_cost`
   - `annual_margin_difference_28_minus_14 = margin_28 - margin_14`
   - Round each currency field to 2 decimal places.

9. **Sort panels** by `panel_code` ascending (standard string sort).

10. **Compute totals**:
    - `total_annual_margin_14_day_usd` = sum of all panels' `annual_margin_14_day_usd`
    - `total_annual_margin_28_day_usd` = sum of all panels' `annual_margin_28_day_usd`
    - `total_annual_margin_difference_28_minus_14_usd` = sum of all per-panel differences
    - `absolute_total_margin_difference_usd` = abs(total_difference)
    - Round each to 2 decimals.

11. **Decision**:
    - If `absolute_total_margin_difference_usd < 6000`: decision = `adopt_28_day`
    - Otherwise: decision = `keep_14_day`
    - Write a justification string that includes the absolute difference and threshold.

12. **Build JSON output**: Load `report_template.json`. Preserve its `metadata` and `audit_notes` exactly. Build the `analysis` section with `assumptions`, `panels`, `totals`, and `recommendation` as specified in the schema. Write to `/root/diagpanel_policy_report.json` with `json.dump(..., indent=2)`.

13. **Build markdown summary**: Write `/root/diagpanel_policy_summary.md` with 4-8 non-empty lines including:
    - Total 14-day margin (USD)
    - Total 28-day margin (USD)
    - Absolute difference (USD)
    - Final decision using exact slug (`adopt_28_day` or `keep_14_day`)

Run the script:
```
python3 /root/solve.py
```

## Step 3: Validate outputs

```
cat /root/diagpanel_policy_report.json
cat /root/diagpanel_policy_summary.md
```

Verify:
- JSON is valid and parseable.
- `metadata` and `audit_notes` match `report_template.json` exactly.
- `analysis.assumptions` matches the specified values.
- `analysis.panels` is sorted by `panel_code` ascending.
- Each panel has all required fields with numeric values rounded to 2 decimals.
- `totals` fields are consistent with summing per-panel values.
- `recommendation.decision` is one of the two exact slugs.
- The markdown summary has 4-8 non-empty lines and includes all required figures and the decision slug.

If any issues are found, fix and re-run.

**IMPORTANT NOTES:**
- When matching `panel_ref` from contract_terms to panels, check both `panel_name` and every element in `alias_labels` (which is a list in the manifest).
- Be careful with data types: parse numeric fields from CSV as float/int appropriately.
- The `shipper_class` field comes from `panel_manifest.json` for each panel.
- The `network_tier` field comes from `panel_manifest.json` for each panel.
- Double-check that `annual_shipper_cost` uses `shipper_cost_usd * runs_per_year` (the shipper cost is per shipment/run, not per lab). Re-read the task: the formula says `annual_margin = annual_revenue - annual_reagent_cost - annual_shipper_cost`. Since revenue and reagent cost both scale with active_labs, but the instructions don't explicitly multiply shipper cost by active_labs, carefully check if shipper_cost should be multiplied by active_labs too. Look at the magnitudes in the data to determine what makes sense. If the shipper_cost CSV has values that look like per-shipment costs (e.g., hundreds of dollars), then `annual_shipper_cost = shipper_cost_usd * runs_per_year` (not multiplied by active_labs) is likely correct. But if the numbers seem too small relative to revenue, consider `shipper_cost_usd * active_labs * runs_per_year`. Use your judgment based on the data, but default to `shipper_cost_usd * runs_per_year` unless the data clearly suggests otherwise.

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