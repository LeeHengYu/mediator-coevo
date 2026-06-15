# Task Instruction

Execute the following steps in order.

## Step 1 – Inspect all input files

Read and display every input file so you understand their schemas before writing any code:

```bash
cat /root/panel_manifest.json
cat /root/shipper_cost.csv
cat /root/contract_terms.csv
cat /root/network_adjustments.csv
cat /root/lab_capacity_overrides.csv
cat /root/holdouts.json
cat /root/report_template.json
```

## Step 2 – Write and run the computation script

Create `/root/solve.py` with the logic below, then run it with `python3 /root/solve.py`.

The script must:

1. **Load inputs**
   - `panel_manifest.json` → list of panel objects
   - `shipper_cost.csv` → CSV; build dict `shipper_class → shipper_cost_usd` (float)
   - `contract_terms.csv` → CSV rows with at least: `panel_ref`, `status_flag`, `effective_week`, `base_payment_per_run_per_lab_usd`
   - `network_adjustments.csv` → CSV; build dict `network_tier → network_adjustment_per_run_per_lab_usd` (float)
   - `lab_capacity_overrides.csv` → CSV rows with at least: `panel_code`, `approval`, `rev`, `active_labs`
   - `holdouts.json` → list; build set of `panel_code` values where `holdout_state == "exclude"`
   - `report_template.json` → preserve `metadata` and `audit_notes` exactly

2. **Filter panels** – keep only manifest entries with `analysis_mode == "review"` AND whose `panel_code` is NOT in the holdout-exclude set.

3. **Resolve contract terms for each retained panel**
   - A contract row matches a panel if `panel_ref` equals the panel's `panel_name` OR `panel_ref` appears in the panel's `alias_labels` list.
   - Keep only rows with `status_flag == "current"`.
   - If multiple current rows match, keep the one with the latest `effective_week` (string comparison is fine if weeks are ISO-formatted; otherwise parse as date).
   - Extract `base_payment_per_run_per_lab_usd` (float).

4. **Network adjustment** – look up `network_adjustment_per_run_per_lab_usd` from the dict by the panel's `network_tier`. Default to `0.0` if tier not found.

5. **Active labs override**
   - From `lab_capacity_overrides.csv`, keep rows where `approval == "approved"` AND `rev` is not blank AND `active_labs` is not blank.
   - Among valid rows for the same `panel_code`, keep the one with the highest numeric `rev`.
   - If no valid override row exists for a panel, use `default_active_labs` from the manifest.
   - `active_labs` must be an integer.

6. **Shipper cost** – look up `shipper_cost_usd` from the dict by the panel's `shipper_class`.

7. **Compute per-panel figures** (all floats, round to 2 decimals at the end):
   - `total_payment_per_run_per_lab_usd = base_payment + network_adjustment`
   - For 14-day: `runs = 26`, `tests_per_lab_per_run = tests_per_lab_per_run_14_day`
   - For 28-day: `runs = 13`, `tests_per_lab_per_run = tests_per_lab_per_run_28_day`
   - `annual_revenue = total_payment_per_run_per_lab_usd * active_labs * runs`
   - `annual_reagent_cost = reagent_cost_per_1000_tests_usd * active_labs * tests_per_lab_per_run * runs / 1000`
   - `annual_shipper_cost = shipper_cost_usd * runs`  (NOTE: the shipper_cost_usd from CSV is per-shipment; annual = cost × runs. Check the CSV to see if there's a per-lab dimension; if `shipper_cost.csv` has a single cost per shipper_class with no lab multiplier, use `shipper_cost_usd * runs`. If the numbers seem off, also try `shipper_cost_usd * active_labs * runs` – pick whichever interpretation the data supports. **Important**: re-read the CSV carefully; if there is any `per_lab` or similar column, factor it in.)
   - `annual_margin = annual_revenue - annual_reagent_cost - annual_shipper_cost`
   - `difference = margin_28 - margin_14`

8. **Sort** panels by `panel_code` ascending.

9. **Totals**:
   - `total_annual_margin_14_day_usd` = sum of all 14-day margins
   - `total_annual_margin_28_day_usd` = sum of all 28-day margins
   - `total_annual_margin_difference_28_minus_14_usd` = sum of per-panel differences
   - `absolute_total_margin_difference_usd` = abs(total_difference)

10. **Decision**:
    - If `absolute_total_margin_difference_usd < 6000` → `adopt_28_day`
    - Else → `keep_14_day`

11. **Round** all USD outputs to 2 decimal places.

12. **Build JSON output** matching the schema exactly. Include `metadata` and `audit_notes` from the template verbatim. Include `assumptions` block with the exact values shown in the schema.

13. **Write** `/root/diagpanel_policy_report.json` (pretty-printed, indent=2).

14. **Write** `/root/diagpanel_policy_summary.md` with 4–8 non-empty lines containing:
    - Total 14-day margin (USD)
    - Total 28-day margin (USD)
    - Absolute difference (USD)
    - Final decision using the exact slug `adopt_28_day` or `keep_14_day`

## Step 3 – Validate outputs

After running the script:

```bash
python3 -c "
import json
with open('/root/diagpanel_policy_report.json') as f:
    r = json.load(f)
print('metadata:', r['metadata'])
print('audit_notes:', r['audit_notes'])
print('num panels:', len(r['analysis']['panels']))
for p in r['analysis']['panels']:
    print(p['panel_code'], p['annual_margin_difference_28_minus_14_usd'])
print('totals:', r['analysis']['totals'])
print('recommendation:', r['analysis']['recommendation'])
"
```

```bash
cat /root/diagpanel_policy_summary.md
```

Verify:
- JSON is valid and matches schema (all keys present).
- Panels are sorted by `panel_code`.
- `metadata` and `audit_notes` match the template exactly.
- Summary has 4–8 non-empty lines and contains the required figures and decision slug.
- Currency values are rounded to 2 decimals.

If anything is wrong, fix and re-run.

## Important notes on shipper cost

The benchmark instruction says `annual_shipper_cost` but does not give an explicit formula with `active_labs`. After reading `shipper_cost.csv`, determine whether the cost is per-shipment (multiply by runs only) or per-lab-per-shipment (multiply by active_labs × runs). Look at column names and values for clues. If ambiguous, try the simpler interpretation first (`shipper_cost_usd * runs_per_year`) since the instruction lists shipper cost separately from per-lab quantities. But if validation shows implausible margins, try the other interpretation.

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