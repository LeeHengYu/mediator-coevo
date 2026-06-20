# Task Instruction

Execute the following steps in order to produce `/root/vaxcrate_analysis.json` and `/root/vaxcrate_summary.md`.

## Step 1 – Inspect all input files

Read and display the full contents of:
- `/root/campaign_manifest.json`
- `/root/crate_cost.csv`
- `/root/billing.csv`
- `/root/location_overrides.csv`
- `/root/suspensions.csv`

Also inspect `/root/tests/` (or any `test_output*.py` / `test_outputs*.py` file in the repo) to understand the verifier's exact expectations.

## Step 2 – Write a Python script `/root/solve.py` that does the following

### 2a – Load data
```python
import json, csv, pathlib, math

manifest = json.loads(pathlib.Path('/root/campaign_manifest.json').read_text())
# manifest is expected to be a list of campaign objects (or dict keyed by campaign_id – inspect first)
```
Load the four CSV files with `csv.DictReader`.

### 2b – Filter campaigns
- Keep only campaigns where `analysis_flag == 'review'`.
- Remove any campaign whose `campaign_id` appears in `suspensions.csv` with `suspension_status == 'hold'`.

### 2c – Resolve billing rows
For each retained campaign, find matching rows in `billing.csv` by checking whether `campaign_label` equals the campaign's `campaign_name` **or** appears in the campaign's `alias_labels` list.
- Keep only rows with `status == 'active'`.
- If multiple active rows match, keep the one with the latest (lexicographically greatest) `cycle_tag`.
- Extract `payment_per_dispatch_per_clinic_usd` from that row (convert to float).

### 2d – Resolve active clinics from location_overrides.csv
For each retained campaign:
- Filter `location_overrides.csv` to rows matching `campaign_id`, `state == 'approved'`, non-blank `revision`, non-blank `active_clinics`.
- Among those, pick the row with the highest numeric `revision`.
- Use its `active_clinics` (int).
- If no qualifying row exists, fall back to `default_active_clinics` from the manifest.

### 2e – Look up crate cost
For each campaign, use `crate_tier` from the manifest to look up `crate_cost_usd` in `crate_cost.csv`.

### 2f – Compute per-campaign numbers
For each model (6-day and 12-day):
```
dispatches_per_year_6  = 60;  days_per_dispatch_6  = 6
dispatches_per_year_12 = 30;  days_per_dispatch_12 = 12

annual_revenue        = payment_per_dispatch_per_clinic_usd * active_clinics * dispatches_per_year
annual_drug_cost      = drug_cost_per_1000_doses_usd * active_clinics * doses_per_day * days_per_dispatch * dispatches_per_year / 1000
annual_crate_cost     = crate_cost_usd * dispatches_per_year
annual_margin         = annual_revenue - annual_drug_cost - annual_crate_cost
```
`annual_margin_difference_12_minus_6_usd = margin_12 - margin_6`

Round every currency value to 2 decimal places.

### 2g – Totals and decision
```
total_margin_6  = sum of all annual_margin_6_day_usd
total_margin_12 = sum of all annual_margin_12_day_usd
total_diff      = total_margin_12 - total_margin_6
abs_diff        = abs(total_diff)
```
Decision: if `abs_diff < 11000` → `move_to_12_day`, else `keep_6_day`.

### 2h – Build output JSON
The output **must** match this exact top-level structure with these exact key names:
```json
{
  "assumptions": {
    "dispatches_per_year_6_day": 60,
    "dispatches_per_year_12_day": 30,
    "days_per_dispatch_6_day": 6,
    "days_per_dispatch_12_day": 12,
    "switch_threshold_usd": 11000,
    "override_rule": "highest numeric approved revision with non-empty active_clinics, else default_active_clinics",
    "suspension_rule": "exclude hold campaigns"
  },
  "campaigns": [ ... sorted by campaign_id ascending ... ],
  "totals": {
    "total_annual_margin_6_day_usd": ...,
    "total_annual_margin_12_day_usd": ...,
    "total_annual_margin_difference_12_minus_6_usd": ...,
    "absolute_total_margin_difference_usd": ...
  },
  "recommendation": {
    "decision": "move_to_12_day" or "keep_6_day",
    "justification": "<short explanation including the absolute difference and threshold>"
  }
}
```
Each campaign object must have **exactly** these keys (no extras, no missing):
- `campaign_id`, `campaign_name`, `active_clinics`
- `drug_cost_per_1000_doses_usd`, `doses_per_day`
- `crate_tier`, `crate_cost_usd`
- `payment_per_dispatch_per_clinic_usd`
- `annual_drug_cost_6_day_usd`, `annual_drug_cost_12_day_usd`
- `annual_crate_cost_6_day_usd`, `annual_crate_cost_12_day_usd`
- `annual_revenue_6_day_usd`, `annual_revenue_12_day_usd`
- `annual_margin_6_day_usd`, `annual_margin_12_day_usd`
- `annual_margin_difference_12_minus_6_usd`

Write to `/root/vaxcrate_analysis.json` with `json.dump(..., indent=2)`.

### 2i – Build summary markdown
Write `/root/vaxcrate_summary.md` with 4–8 non-empty lines that include:
- Total 6-day margin (USD) with the number
- Total 12-day margin (USD) with the number
- Absolute difference (USD) with the number
- The exact decision slug `move_to_12_day` or `keep_6_day`

## Step 3 – Run the script
```bash
cd /root && python solve.py
```

## Step 4 – Validate outputs
- `cat /root/vaxcrate_analysis.json` and verify the top-level keys are `assumptions`, `campaigns`, `totals`, `recommendation`.
- `cat /root/vaxcrate_summary.md` and verify it has 4–8 non-empty lines with the required content.
- Check that `recommendation` is a dict with `decision` and `justification` keys.
- Check that `totals` has exactly the four expected keys.
- Check that each campaign object has exactly the 17 expected keys listed above.

## Step 5 – Run verifier tests if available
```bash
cd /root && python -m pytest tests/ -v 2>&1 | head -80
```
If any test fails, read the error, fix the script, re-run, and re-validate. Do not weaken or modify any test files.

## Critical reminders from previous failure
- `decision` and `justification` must be INSIDE a `recommendation` dict, NOT at top level.
- Use the exact key names from the schema (e.g., `total_annual_margin_6_day_usd`, NOT `total_margin_6`).
- Every per-campaign key must match the schema exactly (e.g., `annual_revenue_6_day_usd`, NOT `annual_revenue_6`).
- `assumptions` must be a dict with the specified keys, NOT a list of strings.
- All currency values rounded to 2 decimal places.
- `crate_cost_usd` in the per-campaign object is the per-crate cost (from the CSV), not the annual crate cost.
- `annual_crate_cost` = `crate_cost_usd * dispatches_per_year` (crate cost is per dispatch, not per clinic).

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
Task metadata: author_email=gpt54@example.com, author_name=GPT-5.4, category=financial-analysis, difficulty=medium, tags=[vaccination, json, csv, distractor-handling, decision-analysis].
Verifier config: timeout_sec=900.0.