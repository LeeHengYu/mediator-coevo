# Task Instruction

Execute the following steps in order to produce `/root/vaxcrate_analysis.json` and `/root/vaxcrate_summary.md`.

## Step 1 — Inspect all input files

Read and print the full contents of each input file:
- `/root/campaign_manifest.json`
- `/root/crate_cost.csv`
- `/root/billing.csv`
- `/root/location_overrides.csv`
- `/root/suspensions.csv`

Also read `/root/tests/test_outputs.py` (or `/tests/test_outputs.py`) to understand the verifier's exact assertions and expected values.

## Step 2 — Write and run a Python script

Create `/root/solve.py` that does the following, then run it with `python /root/solve.py`.

### 2a. Load data
```python
import json, csv, math

with open('/root/campaign_manifest.json') as f:
    manifest = json.load(f)

def read_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))

crate_cost_rows = read_csv('/root/crate_cost.csv')
billing_rows = read_csv('/root/billing.csv')
override_rows = read_csv('/root/location_overrides.csv')
suspension_rows = read_csv('/root/suspensions.csv')
```

### 2b. Build lookup structures
- `crate_cost_map`: dict mapping `crate_tier` → `crate_cost_usd` (float)
- `suspended_ids`: set of `campaign_id` values from suspensions.csv where `suspension_status` stripped/lowered == `hold`

### 2c. Filter campaigns
- From the manifest's campaigns list (it may be a list or dict — inspect the structure), keep only those with `analysis_flag == 'review'`.
- Then exclude any whose `campaign_id` is in `suspended_ids`.
- Store the remaining campaigns in a list.

### 2d. Resolve billing
For each retained campaign:
- The campaign has `campaign_name` and possibly `alias_labels` (a list of strings).
- A billing row matches if its `campaign_label` equals `campaign_name` OR any entry in `alias_labels`.
- Keep only billing rows with `status` == `active` (compare stripped/lowered).
- If multiple active rows match, keep the one with the latest `cycle_tag` (compare as strings — they are typically formatted like `YYYY-QN` or similar; lexicographic comparison works if format is consistent, but inspect the data first).
- Extract `payment_per_dispatch_per_clinic_usd` (float) from the retained billing row.

### 2e. Resolve active clinics from location_overrides
For each retained campaign:
- Filter `location_overrides.csv` rows where `campaign_id` matches AND `state` stripped/lowered == `approved`.
- Among those, discard rows where `revision` is blank/empty (after stripping) OR `active_clinics` is blank/empty (after stripping).
- If multiple valid rows remain, keep the one with the highest numeric `revision` (convert to int or float for comparison).
- Use `active_clinics` (int) from that row.
- If NO valid override row exists, use `default_active_clinics` from the manifest entry for that campaign.

### 2f. Compute per-campaign financials
For each retained campaign, using these values:
- `drug_cost_per_1000_doses_usd` from manifest
- `doses_per_day` from manifest
- `crate_tier` from manifest
- `crate_cost_usd` from `crate_cost_map[crate_tier]`
- `payment_per_dispatch_per_clinic_usd` from billing
- `active_clinics` from step 2e

Compute:
```
# 6-day model
annual_revenue_6 = payment_per_dispatch_per_clinic_usd * active_clinics * 60
annual_drug_cost_6 = drug_cost_per_1000_doses_usd * active_clinics * doses_per_day * 6 * 60 / 1000
annual_crate_cost_6 = crate_cost_usd * 60
annual_margin_6 = annual_revenue_6 - annual_drug_cost_6 - annual_crate_cost_6

# 12-day model
annual_revenue_12 = payment_per_dispatch_per_clinic_usd * active_clinics * 30
annual_drug_cost_12 = drug_cost_per_1000_doses_usd * active_clinics * doses_per_day * 12 * 30 / 1000
annual_crate_cost_12 = crate_cost_usd * 30
annual_margin_12 = annual_revenue_12 - annual_drug_cost_12 - annual_crate_cost_12

difference = annual_margin_12 - annual_margin_6
```

**CRITICAL**: Note that `annual_crate_cost` is `crate_cost_usd * dispatches_per_year` (NOT multiplied by `active_clinics`). The crate cost is per dispatch, not per clinic per dispatch. Verify this interpretation against the test expectations. If the test file reveals that crate cost IS per clinic, adjust accordingly. Read the test file carefully.

**ALSO CRITICAL**: Double-check that `annual_drug_cost` uses `days_per_dispatch` (6 or 12), NOT `dispatches_per_year` alone. The formula is:
`drug_cost_per_1000_doses_usd * active_clinics * doses_per_day * days_per_dispatch * dispatches_per_year / 1000`

Note that the drug cost formula has 5 multiplicands plus division by 1000. Make sure all 5 are present.

### 2g. Round all currency values to 2 decimal places
Use `round(value, 2)` for every USD output field.

### 2h. Sort campaigns by campaign_id ascending

### 2i. Compute totals
```
total_margin_6 = sum of all annual_margin_6
total_margin_12 = sum of all annual_margin_12
total_difference = sum of all per-campaign differences  (equivalently total_margin_12 - total_margin_6)
absolute_difference = abs(total_difference)
```
Round each to 2 decimals.

### 2j. Decision
- If `abs(total_difference) < 11000`: decision = `move_to_12_day`
- Otherwise: decision = `keep_6_day`

### 2k. Write JSON output
Write `/root/vaxcrate_analysis.json` with the exact schema from the task. Include the `assumptions` block with the exact keys and values shown. Include a `justification` string in the recommendation that mentions the absolute difference and the threshold.

### 2l. Write markdown summary
Write `/root/vaxcrate_summary.md` with 4–8 non-empty lines including:
- Total 6-day margin in USD
- Total 12-day margin in USD
- Absolute difference in USD
- Final decision using exact slug (`move_to_12_day` or `keep_6_day`)

## Step 3 — Run the verifier

Run: `cd /root && python -m pytest tests/test_outputs.py -v` (or wherever the test file is located — check `/root/tests/` and `/tests/`).

If any test fails:
1. Read the exact assertion error and expected values.
2. Print diagnostic info: the campaign data, intermediate calculations, active_clinics values, billing matches, crate costs.
3. Identify the discrepancy and fix the calculation in solve.py.
4. Re-run the script and re-run the tests.
5. Repeat until all tests pass.

## Important debugging notes from prior failure

The previous run had these failures:
- `close(468.0, 7956.0)` — a per-campaign value was off by roughly 17x. This strongly suggests either `active_clinics` was wrong (e.g., using 1 instead of 17) or the crate cost formula was wrong (missing multiplication by clinics or dispatches).
- `close(-45894.84, -83406.84)` — total margin was off by ~$37,500, consistent with systematic per-campaign errors.

Pay special attention to:
1. Whether `annual_crate_cost` should be `crate_cost_usd * dispatches_per_year` or `crate_cost_usd * active_clinics * dispatches_per_year`. Check the test file for clues.
2. Whether `active_clinics` resolution is picking the right override row.
3. Whether billing row matching handles `alias_labels` correctly (it's a list in the manifest, not a string).
4. Whether `cycle_tag` comparison for "latest" is lexicographic or needs date parsing.

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