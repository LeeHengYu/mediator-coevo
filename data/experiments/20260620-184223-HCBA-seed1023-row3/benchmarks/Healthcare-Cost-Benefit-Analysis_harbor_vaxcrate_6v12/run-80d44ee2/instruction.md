# Task Instruction

Execute the following steps in order to produce `/root/vaxcrate_analysis.json` and `/root/vaxcrate_summary.md`.

## Step 1 – Inspect all input files

Read and display the full contents of:
- `/root/campaign_manifest.json`
- `/root/crate_cost.csv`
- `/root/billing.csv`
- `/root/location_overrides.csv`
- `/root/suspensions.csv`

Also read `/root/test_output.py` (or any test/verifier script in `/root/`) so you know exactly what assertions the verifier makes.

## Step 2 – Write a single Python script `/root/solve.py` that does everything below, then run it.

### 2a – Load data
```python
import json, csv, pathlib, math

with open('/root/campaign_manifest.json') as f:
    manifest = json.load(f)

def read_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))

crate_cost_rows = read_csv('/root/crate_cost.csv')
billing_rows    = read_csv('/root/billing.csv')
override_rows   = read_csv('/root/location_overrides.csv')
suspension_rows = read_csv('/root/suspensions.csv')
```

### 2b – Build lookup structures
- `crate_cost_map`: `{crate_tier: float(crate_cost_usd)}` from `crate_cost.csv`.
- `suspended_ids`: set of `campaign_id` where `suspension_status` == `"hold"` (strip & lowercase compare).

### 2c – Filter campaigns
From `manifest["campaigns"]` (it may be a list or a dict keyed by id – inspect the actual structure):
- Keep only those with `analysis_flag == "review"`.
- Exclude any whose `campaign_id` is in `suspended_ids`.

### 2d – Resolve billing
For each retained campaign:
1. Collect all rows from `billing.csv` where `status` == `"active"` AND `campaign_label` matches either `campaign_name` or any element of `alias_labels` (which is a list in the manifest; be sure to handle both string and list types).
2. Among matching active rows, keep the one with the **latest** `cycle_tag` (compare as strings – they are typically in YYYY-MM or similar sortable format; inspect the actual values).
3. Extract `payment_per_dispatch_per_clinic_usd` as a float.

### 2e – Resolve active clinics from location_overrides
For each retained campaign:
1. Filter `location_overrides.csv` to rows matching `campaign_id`, `state` == `"approved"`.
2. Drop rows where `revision` is blank/empty or `active_clinics` is blank/empty.
3. Among remaining, keep the row with the **highest numeric `revision`** (convert to int or float for comparison).
4. Use its `active_clinics` (convert to int).
5. If no valid override row exists, fall back to `default_active_clinics` from the manifest entry.

### 2f – Compute per-campaign numbers
Constants:
```
days_6, disp_6 = 6, 60
days_12, disp_12 = 12, 30
```

For each campaign pull from manifest:
- `drug_cost_per_1000_doses_usd` (float)
- `doses_per_day` (float)
- `crate_tier` (string)

Look up `crate_cost_usd` from `crate_cost_map[crate_tier]`.

**CRITICAL – crate cost is per-dispatch, so annual crate cost = crate_cost_usd × dispatches_per_year.**

Formulas (use the exact names from the schema):
```
annual_revenue_X        = payment_per_dispatch_per_clinic_usd * active_clinics * dispatches_per_year_X
annual_drug_cost_X      = drug_cost_per_1000_doses_usd * active_clinics * doses_per_day * days_per_dispatch_X * dispatches_per_year_X / 1000
annual_crate_cost_X     = crate_cost_usd * dispatches_per_year_X
annual_margin_X         = annual_revenue_X - annual_drug_cost_X - annual_crate_cost_X
```

`annual_margin_difference_12_minus_6 = annual_margin_12 - annual_margin_6`

Round every currency value to 2 decimal places using Python's `round(..., 2)`.

### 2g – Totals and decision
```
total_margin_6  = sum of annual_margin_6_day_usd across campaigns
total_margin_12 = sum of annual_margin_12_day_usd across campaigns
total_diff      = total_margin_12 - total_margin_6
abs_diff        = abs(total_diff)
```
Round each to 2 decimals.

Decision rule:
- If `abs_diff < 11000` → `"move_to_12_day"`
- Otherwise → `"keep_6_day"`

### 2h – Build and write JSON
Sort the campaigns list by `campaign_id` ascending.

Write `/root/vaxcrate_analysis.json` with the exact schema from the task (use `json.dump` with `indent=2`).

The `assumptions` block must be exactly:
```json
{
  "dispatches_per_year_6_day": 60,
  "dispatches_per_year_12_day": 30,
  "days_per_dispatch_6_day": 6,
  "days_per_dispatch_12_day": 12,
  "switch_threshold_usd": 11000,
  "override_rule": "highest numeric approved revision with non-empty active_clinics, else default_active_clinics",
  "suspension_rule": "exclude hold campaigns"
}
```

### 2i – Build and write summary markdown
Write `/root/vaxcrate_summary.md` with 4-8 non-empty lines containing:
- Total 6-day margin in USD (as a plain number with 2 decimals, no `$` prefix – e.g., `Total 6-day margin: -83406.84 USD`)
- Total 12-day margin in USD
- Absolute difference in USD
- The exact decision slug (`move_to_12_day` or `keep_6_day`)

Do NOT prefix numbers with `$`. Use plain numeric format with 2 decimal places.

## Step 3 – Run the script
```bash
cd /root && python solve.py
```

## Step 4 – Validate outputs
- `cat /root/vaxcrate_analysis.json` and verify it parses, has the right keys, campaigns sorted by id.
- `cat /root/vaxcrate_summary.md` and verify 4-8 non-empty lines with required info.
- If a test script exists (`test_output.py` or similar), run it: `cd /root && python -m pytest test_output.py -v` and report results.

## Step 5 – Debug if needed
If any test fails:
1. Read the exact assertion error.
2. Re-inspect the input files for the specific campaign/value that failed.
3. Trace the calculation by hand for that campaign.
4. Fix `solve.py` and re-run.

Repeat until all tests pass or you are confident the outputs are correct.

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