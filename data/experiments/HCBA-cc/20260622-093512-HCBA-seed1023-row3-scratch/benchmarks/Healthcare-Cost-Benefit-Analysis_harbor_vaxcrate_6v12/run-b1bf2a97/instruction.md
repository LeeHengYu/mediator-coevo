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
- From `suspensions.csv`, collect all `campaign_id` values where `suspension_status` == `"hold"`. Exclude those campaigns.

### 2c. Resolve billing
- For each retained campaign, find rows in `billing.csv` where `campaign_label` matches either the campaign's `campaign_name` OR any entry in its `alias_labels` list.
- Keep only billing rows with `status` == `"active"`.
- If multiple active rows match the same campaign, keep the one with the latest (lexicographically greatest) `cycle_tag`.
- Extract `payment_per_dispatch_per_clinic_usd` from that row.

### 2d. Resolve active clinics
- From `location_overrides.csv`, keep rows where `state` == `"approved"`.
- Discard rows where `revision` is blank/empty or `active_clinics` is blank/empty.
- Among remaining rows for a given `campaign_id`, keep the one with the highest numeric `revision`.
- Use that row's `active_clinics`.
- If no valid override row exists for a campaign, use `default_active_clinics` from the manifest.

### 2e. Look up crate cost
- Match each campaign's `crate_tier` against `crate_cost.csv` to get `crate_cost_usd`.

### 2f. Compute per-campaign values
For each campaign, using these constants:
- 6-day: days_per_dispatch=6, dispatches_per_year=60
- 12-day: days_per_dispatch=12, dispatches_per_year=30

Compute:
- `annual_revenue = payment_per_dispatch_per_clinic_usd * active_clinics * dispatches_per_year`
- `annual_drug_cost = drug_cost_per_1000_doses_usd * active_clinics * doses_per_day * days_per_dispatch * dispatches_per_year / 1000`
- `annual_crate_cost = crate_cost_usd * dispatches_per_year`  (NOTE: re-check the data — if crate_cost.csv has a per-dispatch cost, multiply by dispatches_per_year; if it's already annual, use as-is. Read the CSV carefully. The field name `crate_cost_usd` likely means per-dispatch cost, so multiply by dispatches_per_year.)
- `annual_margin = annual_revenue - annual_drug_cost - annual_crate_cost`
- `difference = margin_12 - margin_6`

Round all currency values to 2 decimal places.

### 2g. Compute totals
- Sum all per-campaign margins for 6-day and 12-day.
- `total_difference = total_margin_12 - total_margin_6`
- `absolute_total_difference = abs(total_difference)`
- Round to 2 decimals.

### 2h. Decision
- If `abs(total_difference) < 11000`, decision = `"move_to_12_day"`
- Otherwise, decision = `"keep_6_day"`
- Write a short justification string mentioning the absolute difference and threshold.

### 2i. Write `/root/vaxcrate_analysis.json`
- Use the exact JSON schema from the task.
- Sort `campaigns` array by `campaign_id` ascending.
- Use `json.dump` with `indent=2`.

### 2j. Write `/root/vaxcrate_summary.md`
- 4–8 non-empty lines.
- Must include: total 6-day margin (USD), total 12-day margin (USD), absolute difference (USD), and the exact decision slug (`move_to_12_day` or `keep_6_day`).

## Step 3: Validate outputs

```
cat /root/vaxcrate_analysis.json
cat /root/vaxcrate_summary.md
python3 -c "import json; d=json.load(open('/root/vaxcrate_analysis.json')); print('campaigns:', len(d['campaigns'])); print('decision:', d['recommendation']['decision']); print('totals:', d['totals'])"
```

Verify:
- JSON is valid and matches the schema.
- All currency values are rounded to 2 decimals.
- Campaigns are sorted by campaign_id ascending.
- Summary has 4-8 non-empty lines with required content.
- The decision slug in the summary matches the JSON decision exactly.

IMPORTANT NOTES:
- Pay very careful attention to the `annual_crate_cost` formula. The task says `crate_cost_usd` from `crate_cost.csv`. Look at the CSV to understand what this cost represents. The task does not give an explicit formula for annual_crate_cost beyond referencing `crate_cost_usd`. Since the task gives formulas for drug cost and revenue that both scale by dispatches_per_year, and since crate cost logically scales per dispatch, multiply `crate_cost_usd * dispatches_per_year`. But if the numbers don't make sense, re-examine.
- When matching `campaign_label` from billing to campaigns, be careful with exact string matching. `alias_labels` in the manifest is likely a list; check if `campaign_label` is in that list OR equals `campaign_name`.
- `doses_per_day` and `drug_cost_per_1000_doses_usd` come from the campaign manifest, not billing.
- Make sure to handle numeric types properly — CSV values may be strings.

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