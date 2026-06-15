# Task Instruction

Execute the following steps in order. Do not skip any step.

## Step 1 – Inspect all input files

```bash
cat /root/campaign_manifest.json
cat /root/crate_cost.csv
cat /root/billing.csv
cat /root/location_overrides.csv
cat /root/suspensions.csv
```

Read every file carefully before writing any code.

## Step 2 – Write and run a Python script

Create `/root/solve.py` with the following logic, then run it with `python3 /root/solve.py`.

### 2a – Load data
- Load `campaign_manifest.json` (a JSON file; could be a single object with a list, or a list of campaign objects — inspect to confirm structure).
- Load `crate_cost.csv`, `billing.csv`, `location_overrides.csv`, `suspensions.csv` as CSVs (use `csv.DictReader`).

### 2b – Filter campaigns
- From the manifest, keep only campaigns where `analysis_flag == "review"`.
- Load `suspensions.csv`. Collect every `campaign_id` whose `suspension_status` equals `hold` (case-sensitive match on the exact string in the file — inspect the file first).
- Remove any campaign whose `campaign_id` is in that hold set.
- The remaining campaigns are the "retained" campaigns.

### 2c – Resolve billing rows
For each retained campaign:
- The campaign has a `campaign_name` and possibly `alias_labels` (a list of strings) in the manifest.
- In `billing.csv`, find rows where `campaign_label` matches either the `campaign_name` or any entry in `alias_labels`.
- Keep only rows where `status` is `active`.
- If multiple active rows match, keep the one with the latest (lexicographically greatest) `cycle_tag`.
- Extract `payment_per_dispatch_per_clinic_usd` from the retained billing row (convert to float).

### 2d – Resolve active clinics from location_overrides
For each retained campaign:
- In `location_overrides.csv`, find rows matching by `campaign_id`.
- Keep only rows where `state` is `approved`.
- Among those, discard rows where `revision` is blank/empty or `active_clinics` is blank/empty.
- If multiple valid rows remain, keep the one with the highest numeric `revision`.
- Use `active_clinics` (convert to int) from that row.
- If no valid override row exists, use `default_active_clinics` from the manifest (convert to int).

### 2e – Look up crate cost
For each retained campaign:
- Get `crate_tier` from the manifest.
- Look up `crate_cost_usd` from `crate_cost.csv` by matching `crate_tier`.
- Convert to float.

### 2f – Compute per-campaign numbers
Constants:
- 6-day: days_per_dispatch=6, dispatches_per_year=60
- 12-day: days_per_dispatch=12, dispatches_per_year=30

For each retained campaign, using manifest fields `drug_cost_per_1000_doses_usd` (float) and `doses_per_day` (float):

```
annual_drug_cost = drug_cost_per_1000_doses_usd * active_clinics * doses_per_day * days_per_dispatch * dispatches_per_year / 1000
annual_crate_cost = crate_cost_usd * dispatches_per_year
annual_revenue = payment_per_dispatch_per_clinic_usd * active_clinics * dispatches_per_year
annual_margin = annual_revenue - annual_drug_cost - annual_crate_cost
```

Compute for both 6-day and 12-day models.

`annual_margin_difference_12_minus_6 = annual_margin_12_day - annual_margin_6_day`

Round every currency value to 2 decimal places.

### 2g – Compute totals
```
total_annual_margin_6_day = sum of all annual_margin_6_day
total_annual_margin_12_day = sum of all annual_margin_12_day
total_annual_margin_difference = sum of all per-campaign differences
absolute_total_margin_difference = abs(total_annual_margin_difference)
```
Round each to 2 decimals.

### 2h – Decision
- If `absolute_total_margin_difference < 11000`, decision is `move_to_12_day`.
- Otherwise, decision is `keep_6_day`.

### 2i – Build and write JSON output
Build the JSON object exactly matching the schema in the task. The `campaigns` array must be sorted by `campaign_id` ascending (string sort). Write to `/root/vaxcrate_analysis.json` with `json.dump(..., indent=2)`.

The `justification` string should be a brief sentence including the absolute difference value and the threshold.

### 2j – Write summary markdown
Write `/root/vaxcrate_summary.md` with 4–8 non-empty lines including:
- Total 6-day margin (USD) with 2-decimal value
- Total 12-day margin (USD) with 2-decimal value
- Absolute difference (USD) with 2-decimal value
- The exact decision slug (`move_to_12_day` or `keep_6_day`)

Example format:
```
# VaxCrate Dispatch Analysis Summary

Total annual margin under 6-day policy: $XXXXX.XX USD
Total annual margin under 12-day policy: $XXXXX.XX USD
Absolute margin difference: $XXXXX.XX USD
Recommendation: <decision_slug>
```

## Step 3 – Validate outputs

```bash
cat /root/vaxcrate_analysis.json
cat /root/vaxcrate_summary.md
python3 -c "import json; d=json.load(open('/root/vaxcrate_analysis.json')); print('campaigns:', len(d['campaigns'])); print('decision:', d['recommendation']['decision']); print('totals:', d['totals'])"
```

Verify:
- JSON is valid and parseable.
- All currency fields are rounded to 2 decimals.
- campaigns array is sorted by campaign_id.
- Summary has 4-8 non-empty lines and contains the required info.
- Decision slug matches between JSON and summary.

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