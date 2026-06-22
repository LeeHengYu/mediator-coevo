# Task Instruction

Execute the following steps in order:

## Step 1: Inspect all input files

Read and display the contents of each input file:
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
- Load `campaign_manifest.json` (list of campaign objects).
- Load `crate_cost.csv`, `billing.csv`, `location_overrides.csv`, `suspensions.csv` using the `csv` module (handle any BOM or encoding issues).

### 2b. Filter campaigns
- Keep only campaigns where `analysis_flag == "review"`.
- From `suspensions.csv`, collect every `campaign_id` whose `suspension_status == "hold"`. Remove those campaigns.

### 2c. Resolve billing
- For each retained campaign, find matching rows in `billing.csv` by checking if the billing row's `campaign_label` equals the campaign's `campaign_name` OR appears in the campaign's `alias_labels` list.
- Keep only billing rows with `status == "active"`.
- If multiple active rows match, keep the one with the latest (lexicographically greatest) `cycle_tag`.
- Extract `payment_per_dispatch_per_clinic_usd` from the retained billing row (convert to float).

### 2d. Resolve active clinics
- From `location_overrides.csv`, keep rows where `state == "approved"` AND `revision` is not blank AND `active_clinics` is not blank.
- Group valid rows by `campaign_id`. For each campaign, keep the row with the highest numeric `revision`.
- For each retained campaign: if a valid override row exists, use its `active_clinics` (as int). Otherwise use `default_active_clinics` from the manifest (as int).

### 2e. Resolve crate cost
- Build a lookup from `crate_cost.csv`: `crate_tier` -> `crate_cost_usd` (float).
- For each campaign, use its `crate_tier` from the manifest to look up the cost.

### 2f. Compute per-campaign financials
For each retained campaign, using these parameters:
- 6-day model: `days_per_dispatch=6`, `dispatches_per_year=60`
- 12-day model: `days_per_dispatch=12`, `dispatches_per_year=30`
- `drug_cost_per_1000_doses_usd` and `doses_per_day` from the manifest (as floats).

Formulas:
- `annual_revenue = payment_per_dispatch_per_clinic_usd * active_clinics * dispatches_per_year`
- `annual_drug_cost = drug_cost_per_1000_doses_usd * active_clinics * doses_per_day * days_per_dispatch * dispatches_per_year / 1000`
- `annual_crate_cost = crate_cost_usd * dispatches_per_year`  (Note: crate cost is per dispatch, so annual = crate_cost_usd * dispatches_per_year. If the data or schema suggests a different interpretation, check the crate_cost.csv structure carefully. The most natural reading is that `crate_cost_usd` is the cost per crate per dispatch.)
- `annual_margin = annual_revenue - annual_drug_cost - annual_crate_cost`
- `difference = annual_margin_12_day - annual_margin_6_day`

Round ALL currency values to 2 decimal places using Python's `round(x, 2)`.

### 2g. Compute totals
- `total_annual_margin_6_day_usd` = sum of all campaigns' `annual_margin_6_day_usd`
- `total_annual_margin_12_day_usd` = sum of all campaigns' `annual_margin_12_day_usd`
- `total_annual_margin_difference_12_minus_6_usd` = sum of all campaigns' `annual_margin_difference_12_minus_6_usd`
- `absolute_total_margin_difference_usd` = `abs(total_annual_margin_difference_12_minus_6_usd)`
- Round all totals to 2 decimals.

### 2h. Decision
- If `absolute_total_margin_difference_usd < 11000`, decision = `move_to_12_day`.
- Otherwise, decision = `keep_6_day`.
- Write a justification string that mentions the absolute difference and the threshold.

### 2i. Build and write JSON
- Sort the campaigns array by `campaign_id` ascending.
- Write `/root/vaxcrate_analysis.json` with the exact schema shown in the task, using `json.dump` with `indent=2`.
- Ensure all numeric currency fields are floats rounded to 2 decimals.

### 2j. Write summary markdown
- Write `/root/vaxcrate_summary.md` with 4-8 non-empty lines containing:
  - Total 6-day margin (USD)
  - Total 12-day margin (USD)
  - Absolute difference (USD)
  - Final decision using the exact slug (`move_to_12_day` or `keep_6_day`)

## Step 3: Validate outputs

After the script runs:
```
cat /root/vaxcrate_analysis.json
cat /root/vaxcrate_summary.md
```

Verify:
- JSON is valid and parseable.
- `assumptions` block matches the required static values exactly.
- `campaigns` array is sorted by `campaign_id`.
- All currency fields are rounded to 2 decimals.
- Summary has 4-8 non-empty lines with the required information and exact decision slug.
- The decision logic is correct: `abs(total_difference) < 11000` → `move_to_12_day`, otherwise `keep_6_day`.

If anything looks wrong, debug and fix before finishing.

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