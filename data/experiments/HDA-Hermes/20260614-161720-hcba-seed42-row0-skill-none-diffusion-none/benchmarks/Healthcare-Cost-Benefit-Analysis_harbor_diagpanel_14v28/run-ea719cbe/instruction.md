# Task Instruction

## Task: Healthcare Cost-Benefit Analysis – Harbor DiagPanel 14v28

You must produce two output files by reading and processing several input files according to precise business rules. Follow every step carefully.

### Step 1: Read all input files

Read these files and inspect their contents:
- `/root/panel_manifest.json`
- `/root/shipper_cost.csv`
- `/root/contract_terms.csv`
- `/root/network_adjustments.csv`
- `/root/lab_capacity_overrides.csv`
- `/root/holdouts.json`
- `/root/report_template.json`

Print/dump each one so you understand the exact field names, data types, and values before writing any code.

### Step 2: Write a Python script `/root/solve.py` that does the following

#### 2a. Determine retained panels
- From `panel_manifest.json`, select panels where `analysis_mode == "review"`.
- From `holdouts.json`, exclude any panel whose `panel_code` has `holdout_state == "exclude"`.
- The remaining panels are the "retained" panels.

#### 2b. Resolve contract terms
- Read `contract_terms.csv`. For each row, match `panel_ref` to a retained panel by checking if `panel_ref` equals the panel's `panel_name` OR if `panel_ref` appears in the panel's `alias_labels` list.
- Keep only rows where `status_flag == "current"` (case-sensitive match on the CSV value).
- If multiple current rows map to the same retained panel, keep only the one with the latest `effective_week` (compare as strings if ISO dates, or parse properly).
- Extract `base_payment_per_run_per_lab_usd` from the winning contract row.

#### 2c. Network adjustment
- Read `network_adjustments.csv`. For each retained panel, look up `network_adjustment_per_run_per_lab_usd` by matching the panel's `network_tier`.
- If the panel's `network_tier` is not found in the CSV, use `0.0`.

#### 2d. Active labs
- Read `lab_capacity_overrides.csv`. Keep only rows where `approval == "approved"` (case-sensitive).
- Among approved rows, discard any row whose `rev` field is blank/empty or whose `active_labs` field is blank/empty.
- Convert `rev` to numeric. If multiple valid approved rows exist for the same `panel_code`, keep the one with the highest numeric `rev`.
- For each retained panel, if a valid override row exists, use its `active_labs` (as integer). Otherwise use `default_active_labs` from `panel_manifest.json`.

#### 2e. Shipper cost
- Read `shipper_cost.csv`. For each retained panel, look up `shipper_cost_usd` by matching the panel's `shipper_class`.

#### 2f. Compute per-panel financials
For each retained panel:
- `total_payment_per_run_per_lab_usd = base_payment_per_run_per_lab_usd + network_adjustment_per_run_per_lab_usd`
- **14-day model** (26 runs/year):
  - `annual_revenue_14 = total_payment_per_run_per_lab_usd * active_labs * 26`
  - `annual_reagent_cost_14 = reagent_cost_per_1000_tests_usd * active_labs * tests_per_lab_per_run_14_day * 26 / 1000`
  - `annual_shipper_cost_14 = shipper_cost_usd * 26`
  - `annual_margin_14 = annual_revenue_14 - annual_reagent_cost_14 - annual_shipper_cost_14`
- **28-day model** (13 runs/year):
  - `annual_revenue_28 = total_payment_per_run_per_lab_usd * active_labs * 13`
  - `annual_reagent_cost_28 = reagent_cost_per_1000_tests_usd * active_labs * tests_per_lab_per_run_28_day * 13 / 1000`
  - `annual_shipper_cost_28 = shipper_cost_usd * 13`
  - `annual_margin_28 = annual_revenue_28 - annual_reagent_cost_28 - annual_shipper_cost_28`
- `annual_margin_difference_28_minus_14 = annual_margin_28 - annual_margin_14`

#### 2g. Totals and decision
- `total_annual_margin_14 = sum of all per-panel annual_margin_14`
- `total_annual_margin_28 = sum of all per-panel annual_margin_28`
- `total_difference = total_annual_margin_28 - total_annual_margin_14`
- `absolute_total_difference = abs(total_difference)`
- Decision: if `absolute_total_difference < 6000` → `adopt_28_day`, else → `keep_14_day`.

#### 2h. Build JSON output
- Read `report_template.json`. Preserve its `metadata` and `audit_notes` exactly as-is.
- Build the `analysis` block per the schema below. Sort `panels` by `panel_code` ascending.
- Round ALL currency values to 2 decimal places.
- Write to `/root/diagpanel_policy_report.json` with `json.dump(..., indent=2)`.

JSON schema for output:
```json
{
  "metadata": { ... from template ... },
  "audit_notes": [ ... from template ... ],
  "analysis": {
    "assumptions": {
      "runs_per_year_14_day": 26,
      "runs_per_year_28_day": 13,
      "switch_threshold_usd": 6000,
      "override_rule": "highest numeric approved rev with non-empty active_labs, else default_active_labs",
      "holdout_rule": "exclude holdout_state=exclude",
      "adjustment_rule": "missing network_tier adjustment defaults to 0.0"
    },
    "panels": [ ... sorted by panel_code ... ],
    "totals": {
      "total_annual_margin_14_day_usd": ...,
      "total_annual_margin_28_day_usd": ...,
      "total_annual_margin_difference_28_minus_14_usd": ...,
      "absolute_total_margin_difference_usd": ...
    },
    "recommendation": {
      "decision": "adopt_28_day" or "keep_14_day",
      "justification": "<a brief sentence explaining the decision based on the threshold rule>"
    }
  }
}
```

Each panel object must have exactly these fields (all currency rounded to 2 decimals):
- `panel_code`, `panel_name`, `active_labs`, `reagent_cost_per_1000_tests_usd`, `network_tier`, `network_adjustment_per_run_per_lab_usd`, `shipper_class`, `shipper_cost_usd`, `base_payment_per_run_per_lab_usd`, `total_payment_per_run_per_lab_usd`, `tests_per_lab_per_run_14_day`, `tests_per_lab_per_run_28_day`, `annual_reagent_cost_14_day_usd`, `annual_reagent_cost_28_day_usd`, `annual_shipper_cost_14_day_usd`, `annual_shipper_cost_28_day_usd`, `annual_revenue_14_day_usd`, `annual_revenue_28_day_usd`, `annual_margin_14_day_usd`, `annual_margin_28_day_usd`, `annual_margin_difference_28_minus_14_usd`

#### 2i. Build markdown summary
Write `/root/diagpanel_policy_summary.md` with 4–8 non-empty lines containing:
- Total 14-day margin (USD)
- Total 28-day margin (USD)
- Absolute difference (USD)
- Final decision using the exact slug `adopt_28_day` or `keep_14_day`

### Step 3: Run the script
```bash
cd /root && python solve.py
```

### Step 4: Validate outputs
- Read `/root/diagpanel_policy_report.json` and verify:
  - `metadata` and `audit_notes` match the template exactly.
  - `panels` are sorted by `panel_code`.
  - All currency fields have exactly 2 decimal places.
  - The `assumptions` block matches the required values exactly.
  - `totals.total_annual_margin_difference_28_minus_14_usd` equals the sum of per-panel differences.
  - `absolute_total_margin_difference_usd` equals `abs(total_annual_margin_difference_28_minus_14_usd)`.
  - Decision is consistent with the threshold rule.
- Read `/root/diagpanel_policy_summary.md` and verify it has 4–8 non-empty lines and includes all required information with the exact decision slug.

If any validation fails, fix and re-run.

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