# Task Instruction

Execute the following steps in order to produce the two required output files.

## Step 1 – Inspect all input files

```bash
cat /root/panel_manifest.json
cat /root/shipper_cost.csv
cat /root/contract_terms.csv
cat /root/network_adjustments.csv
cat /root/lab_capacity_overrides.csv
cat /root/holdouts.json
cat /root/report_template.json
```

Read every file carefully before writing any code.

## Step 2 – Write and run the Python script

Create `/root/solve.py` with the logic below. After writing it, run `python3 /root/solve.py` and inspect the outputs.

### Logic specification (implement exactly):

1. **Load inputs**
   - `panel_manifest.json` → list of panel objects.
   - `shipper_cost.csv` → rows with at least `shipper_class`, `shipper_cost_usd`.
   - `contract_terms.csv` → rows with at least `panel_ref`, `status_flag`, `effective_week`, `base_payment_per_run_per_lab_usd`.
   - `network_adjustments.csv` → rows with `network_tier`, `network_adjustment_per_run_per_lab_usd`.
   - `lab_capacity_overrides.csv` → rows with `panel_code`, `approval`, `rev`, `active_labs`.
   - `holdouts.json` → list of holdout objects with `panel_code` and `holdout_state`.
   - `report_template.json` → object with `metadata` and `audit_notes` (preserve verbatim).

2. **Filter panels**
   - Keep only panels where `analysis_mode == "review"`.
   - Build a set of excluded panel codes from holdouts where `holdout_state == "exclude"`.
   - Remove any panel whose `panel_code` is in the excluded set.
   - The remaining panels are "retained panels".

3. **Resolve contract terms**
   - For each contract row, keep only rows where `status_flag == "current"` (case-sensitive match as found in file; check the actual file for casing).
   - For each retained panel, find contract rows whose `panel_ref` matches either `panel_name` or any entry in `alias_labels` (which is a list in the manifest).
   - If multiple current rows match one panel, keep the one with the latest `effective_week` (compare as strings if they look like ISO dates, or parse them).
   - Extract `base_payment_per_run_per_lab_usd` (float).

4. **Network adjustment**
   - Build a dict from `network_adjustments.csv`: `network_tier` → `network_adjustment_per_run_per_lab_usd` (float).
   - For each retained panel, look up its `network_tier` from the manifest. If the tier is not in the dict, use `0.0`.

5. **Active labs**
   - From `lab_capacity_overrides.csv`, keep rows where `approval == "approved"` (check actual casing in file).
   - Among those, discard rows where `rev` is blank/empty or `active_labs` is blank/empty.
   - For each `panel_code`, keep the row with the highest numeric `rev`.
   - For each retained panel: if an approved valid override row exists, use its `active_labs` (int). Otherwise use `default_active_labs` from the manifest (int).

6. **Shipper cost**
   - Build a dict from `shipper_cost.csv`: `shipper_class` → `shipper_cost_usd` (float).
   - Each panel has a `shipper_class` in the manifest.

7. **Calculations per panel** (use Python floats, round to 2 decimals at the end)

   For each retained panel:
   ```
   total_payment_per_run_per_lab = base_payment + network_adjustment
   
   annual_revenue_14 = total_payment_per_run_per_lab * active_labs * 26
   annual_revenue_28 = total_payment_per_run_per_lab * active_labs * 13
   
   annual_reagent_cost_14 = reagent_cost_per_1000_tests * active_labs * tests_per_lab_per_run_14_day * 26 / 1000
   annual_reagent_cost_28 = reagent_cost_per_1000_tests * active_labs * tests_per_lab_per_run_28_day * 13 / 1000
   
   annual_shipper_cost_14 = shipper_cost_usd * active_labs * 26
   annual_shipper_cost_28 = shipper_cost_usd * active_labs * 13
   
   annual_margin_14 = annual_revenue_14 - annual_reagent_cost_14 - annual_shipper_cost_14
   annual_margin_28 = annual_revenue_28 - annual_reagent_cost_28 - annual_shipper_cost_28
   
   margin_diff = annual_margin_28 - annual_margin_14
   ```

   **IMPORTANT**: The shipper cost formula is `shipper_cost_usd * active_labs * runs_per_year`. This was the likely source of the previous calculation discrepancy. Do NOT omit `active_labs` from the shipper cost formula.

8. **Build panel output objects** with these exact flat keys (no nesting):
   ```
   panel_code, panel_name, active_labs, reagent_cost_per_1000_tests_usd,
   network_tier, network_adjustment_per_run_per_lab_usd, shipper_class,
   shipper_cost_usd, base_payment_per_run_per_lab_usd,
   total_payment_per_run_per_lab_usd,
   tests_per_lab_per_run_14_day, tests_per_lab_per_run_28_day,
   annual_reagent_cost_14_day_usd, annual_reagent_cost_28_day_usd,
   annual_shipper_cost_14_day_usd, annual_shipper_cost_28_day_usd,
   annual_revenue_14_day_usd, annual_revenue_28_day_usd,
   annual_margin_14_day_usd, annual_margin_28_day_usd,
   annual_margin_difference_28_minus_14_usd
   ```
   All USD values rounded to 2 decimal places. `active_labs` as int. `tests_per_lab_per_run_*` as int (or number as found in manifest).

9. **Sort** the panels list by `panel_code` ascending (standard string sort).

10. **Totals**
    ```
    total_annual_margin_14_day_usd = sum of all panels' annual_margin_14_day_usd (round to 2)
    total_annual_margin_28_day_usd = sum of all panels' annual_margin_28_day_usd (round to 2)
    total_annual_margin_difference_28_minus_14_usd = total_28 - total_14 (round to 2)
    absolute_total_margin_difference_usd = abs(total_difference) (round to 2)
    ```

11. **Decision**
    - If `absolute_total_margin_difference_usd < 6000` → `"adopt_28_day"`
    - Otherwise → `"keep_14_day"`
    - `justification`: a short sentence including the absolute difference and the threshold.

12. **Assumptions block** – use these EXACT strings:
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

13. **Assemble JSON output** `/root/diagpanel_policy_report.json`:
    - `metadata` and `audit_notes` copied verbatim from `report_template.json`.
    - `analysis` with `assumptions`, `panels`, `totals`, `recommendation`.
    - Write with `json.dump(..., indent=2)`.

14. **Write summary** `/root/diagpanel_policy_summary.md`:
    - 4–8 non-empty lines.
    - Must include: total 14-day margin with comma formatting (e.g., `1,234.56`), total 28-day margin, absolute difference, and the exact decision slug (`adopt_28_day` or `keep_14_day`).
    - Example format:
      ```
      # Diagnostics Panel Policy Summary

      Total 14-day annual margin: $X,XXX.XX
      Total 28-day annual margin: $X,XXX.XX
      Absolute margin difference: $X,XXX.XX
      Decision: adopt_28_day
      ```

## Step 3 – Validate outputs

After running the script:
```bash
python3 -c "
import json
with open('/root/diagpanel_policy_report.json') as f:
    d = json.load(f)
assert 'metadata' in d
assert 'audit_notes' in d
a = d['analysis']
assert a['assumptions']['holdout_rule'] == 'exclude holdout_state=exclude'
assert a['assumptions']['override_rule'] == 'highest numeric approved rev with non-empty active_labs, else default_active_labs'
assert a['assumptions']['adjustment_rule'] == 'missing network_tier adjustment defaults to 0.0'
for p in a['panels']:
    assert 'annual_revenue_14_day_usd' in p, f'Missing key in {p[\"panel_code\"]}'
    assert 'network_tier' in p
    assert 'annual_margin_difference_28_minus_14_usd' in p
print('Panels:', [p['panel_code'] for p in a['panels']])
print('Sorted check:', a['panels'] == sorted(a['panels'], key=lambda x: x['panel_code']))
print('Totals:', a['totals'])
print('Decision:', a['recommendation']['decision'])
print('JSON OK')
"

cat /root/diagpanel_policy_summary.md
wc -l /root/diagpanel_policy_summary.md
```

If any check fails, read the error, fix the script, and re-run until all checks pass.

## Step 4 – Run the verifier if available

```bash
if [ -f /root/test_output.py ]; then cd /root && python3 -m pytest test_output.py -v; fi
```

If any test fails, read the failure message carefully, fix the issue in solve.py, and re-run until all tests pass.

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