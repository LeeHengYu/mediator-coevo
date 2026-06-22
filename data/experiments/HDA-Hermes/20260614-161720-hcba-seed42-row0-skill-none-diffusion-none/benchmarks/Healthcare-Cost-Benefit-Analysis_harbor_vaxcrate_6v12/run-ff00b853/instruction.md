# Task Instruction

Execute the following steps in order:

## Step 1: Inspect all input files

Read and display the full contents of each input file:
```
cat /root/campaign_manifest.json
cat /root/crate_cost.csv
cat /root/billing.csv
cat /root/location_overrides.csv
cat /root/suspensions.csv
```

## Step 2: Write and run a Python script to produce both output files

Create `/root/solve.py` with the following logic, then run it with `python3 /root/solve.py`.

The script must:

### 2a. Load data
- Load `campaign_manifest.json` (a JSON array or object containing campaign records).
- Load `crate_cost.csv`, `billing.csv`, `location_overrides.csv`, `suspensions.csv` as CSVs.

### 2b. Filter campaigns
- Keep only campaigns where `analysis_flag` == `"review"`.
- From `suspensions.csv`, collect all `campaign_id` values where `suspension_status` == `"hold"`. Exclude those campaign_ids.

### 2c. Resolve billing
- For each retained campaign, find rows in `billing.csv` where `campaign_label` matches either the campaign's `campaign_name` OR any entry in the campaign's `alias_labels` list.
- Keep only billing rows with `status` == `"active"`.
- If multiple active rows match, keep the one with the latest (lexicographically largest) `cycle_tag`.
- Extract `payment_per_dispatch_per_clinic_usd` from the kept billing row.

### 2d. Resolve active clinics
- From `location_overrides.csv`, keep rows where `state` == `"approved"`.
- Among those, discard rows where `revision` is blank/empty or `active_clinics` is blank/empty.
- For each `campaign_id`, if multiple valid rows exist, keep the one with the highest numeric `revision`.
- For each retained campaign: if a valid override row exists, use its `active_clinics`; otherwise use `default_active_clinics` from the manifest.

### 2e. Look up crate cost
- For each campaign, use its `crate_tier` to look up `crate_cost_usd` from `crate_cost.csv`.

### 2f. Compute per-campaign financials
For each campaign, compute for both 6-day (60 dispatches/year, 6 days/dispatch) and 12-day (30 dispatches/year, 12 days/dispatch) models:

- `annual_revenue = payment_per_dispatch_per_clinic_usd * active_clinics * dispatches_per_year`
- `annual_drug_cost = drug_cost_per_1000_doses_usd * active_clinics * doses_per_day * days_per_dispatch * dispatches_per_year / 1000`
- `annual_crate_cost = crate_cost_usd * dispatches_per_year`  (Note: crate cost is per dispatch, so annual = crate_cost_usd * dispatches_per_year. However, re-check: if crate_cost.csv has a per-dispatch cost, multiply by dispatches_per_year. If it's already annual, just use it. Inspect the data to determine the correct interpretation. The task says "Crate cost uses crate_cost_usd from crate_cost.csv" — this likely means the cost per crate per dispatch. So annual_crate_cost = crate_cost_usd * dispatches_per_year.)
- `annual_margin = annual_revenue - annual_drug_cost - annual_crate_cost`
- `difference = annual_margin_12_day - annual_margin_6_day`

Round all currency values to 2 decimal places.

### 2g. Compute totals and decision
- Sum all per-campaign margins for 6-day and 12-day.
- `total_difference = total_12_day_margin - total_6_day_margin`
- `absolute_total_margin_difference = abs(total_difference)`, rounded to 2 decimals.
- If `abs(total_difference) < 11000`, decision = `"move_to_12_day"`, else `"keep_6_day"`.

### 2h. Sort campaigns by `campaign_id` ascending.

### 2i. Write `/root/vaxcrate_analysis.json`
Write the JSON with the exact schema from the task, using `json.dump` with `indent=2`. Include:
- `assumptions` object with the fixed values.
- `campaigns` array sorted by campaign_id.
- `totals` object.
- `recommendation` object with `decision` and a short `justification` string.

### 2j. Write `/root/vaxcrate_summary.md`
Write 4-8 non-empty lines including:
- Total 6-day margin (USD)
- Total 12-day margin (USD)
- Absolute difference (USD)
- Final decision using exact slug (`move_to_12_day` or `keep_6_day`)

## Step 3: Validate outputs

```
cat /root/vaxcrate_analysis.json
cat /root/vaxcrate_summary.md
python3 -c "import json; d=json.load(open('/root/vaxcrate_analysis.json')); print('campaigns:', len(d['campaigns'])); print('decision:', d['recommendation']['decision']); print('totals:', d['totals'])"
```

Verify:
- JSON is valid and parseable.
- All currency fields are rounded to 2 decimal places.
- campaigns array is sorted by campaign_id ascending.
- Summary has 4-8 non-empty lines and contains all required info with exact slug.
- The decision logic: abs(total_difference) < 11000 → move_to_12_day, otherwise keep_6_day.

IMPORTANT NOTES:
- When matching billing rows, be careful: `alias_labels` in the manifest may be a list of strings. Check if `campaign_label` from billing matches `campaign_name` OR is contained in the `alias_labels` list.
- Ensure numeric conversions are correct (some CSV fields may be strings).
- `crate_cost_usd` from crate_cost.csv: match on the campaign's `crate_tier` field from the manifest.
- The annual_crate_cost formula: carefully inspect the data. The task says "Crate cost uses crate_cost_usd from crate_cost.csv, matched by crate_tier" and the annual margin is revenue - drug_cost - crate_cost. The crate_cost_usd is likely per-dispatch, so multiply by dispatches_per_year. If after inspecting the CSV it appears to be a flat annual cost, adjust accordingly. The most natural reading is per-dispatch.
- Use `< 11000` strictly (not `<=`).
- All rounding to 2 decimal places should use Python's `round(value, 2)`.

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