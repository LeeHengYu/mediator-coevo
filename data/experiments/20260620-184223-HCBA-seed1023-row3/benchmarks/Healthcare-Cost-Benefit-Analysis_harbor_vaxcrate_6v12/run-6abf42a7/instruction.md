# Task Instruction

Execute the following steps in order:

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

Create `/root/solve.py` with the logic below, then run it with `python3 /root/solve.py`.

The script must:

### A. Load data
- Load `campaign_manifest.json` (list of campaign objects).
- Load `crate_cost.csv`, `billing.csv`, `location_overrides.csv`, `suspensions.csv` as CSV.

### B. Filter campaigns
1. Keep only campaigns where `analysis_flag == "review"`.
2. Collect campaign_ids from `suspensions.csv` where `suspension_status == "hold"`. Remove any campaign whose `campaign_id` is in that set.

### C. Resolve billing
For each retained campaign:
- Match `billing.csv` rows where `campaign_label` equals the campaign's `campaign_name` **or** any entry in the campaign's `alias_labels` list.
- Keep only rows with `status == "active"`.
- If multiple active rows match, keep the one with the latest (lexicographically greatest) `cycle_tag`.
- Extract `payment_per_dispatch_per_clinic_usd` from the kept row.

### D. Resolve active clinics
For each retained campaign:
- From `location_overrides.csv`, filter rows matching the campaign's `campaign_id` where `state == "approved"`.
- Drop rows where `revision` is blank/empty or `active_clinics` is blank/empty.
- If valid rows remain, keep the one with the highest numeric `revision` and use its `active_clinics` (as int).
- Otherwise, use `default_active_clinics` from the campaign manifest.

### E. Look up crate cost
For each campaign, use `crate_tier` from the manifest to look up `crate_cost_usd` in `crate_cost.csv`.

### F. Compute per-campaign numbers
For each model (6-day: days=6, dispatches=60; 12-day: days=12, dispatches=30):
- `annual_revenue = payment_per_dispatch_per_clinic_usd * active_clinics * dispatches_per_year`
- `annual_drug_cost = drug_cost_per_1000_doses_usd * active_clinics * doses_per_day * days_per_dispatch * dispatches_per_year / 1000`
- `annual_crate_cost = crate_cost_usd * dispatches_per_year`  (crate cost is per dispatch, so multiply by number of dispatches)
  **IMPORTANT**: Re-read `crate_cost.csv` carefully. If the CSV has a column that already represents annual or per-dispatch cost, use it correctly. The formula is: `crate_cost_usd * dispatches_per_year`. Verify this by inspecting the CSV header and values.
- `annual_margin = annual_revenue - annual_drug_cost - annual_crate_cost`
- `difference = margin_12 - margin_6`

Round every currency value to 2 decimal places.

### G. Totals and decision
- Sum all per-campaign margins for 6-day and 12-day.
- `total_difference = total_margin_12 - total_margin_6`
- `absolute_total_margin_difference = abs(total_difference)`, rounded to 2 decimals.
- If `abs(total_difference) < 11000` → decision = `move_to_12_day`, else `keep_6_day`.

### H. Write `/root/vaxcrate_analysis.json`
Use **exactly** this schema (no extra keys, no missing keys, no nesting beyond what's shown):

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
    "justification": "<brief string>"
  }
}
```

Each campaign object must have exactly these keys (no extras):
`campaign_id`, `campaign_name`, `active_clinics`, `drug_cost_per_1000_doses_usd`, `doses_per_day`, `crate_tier`, `crate_cost_usd`, `payment_per_dispatch_per_clinic_usd`, `annual_drug_cost_6_day_usd`, `annual_drug_cost_12_day_usd`, `annual_crate_cost_6_day_usd`, `annual_crate_cost_12_day_usd`, `annual_revenue_6_day_usd`, `annual_revenue_12_day_usd`, `annual_margin_6_day_usd`, `annual_margin_12_day_usd`, `annual_margin_difference_12_minus_6_usd`

### I. Write `/root/vaxcrate_summary.md`
4-8 non-empty lines. Must include:
- Total 6-day margin formatted with commas, e.g., `-83,406.84` (use Python `f'{value:,.2f}'`)
- Total 12-day margin formatted with commas
- Absolute difference formatted with commas
- The exact decision slug `move_to_12_day` or `keep_6_day`

**Do NOT prefix currency values with `$` in the summary** – just include the raw comma-formatted number so the verifier regex can find it.

## Step 3 – Validate outputs

```bash
python3 -c "
import json
data = json.load(open('/root/vaxcrate_analysis.json'))
assert set(data['assumptions'].keys()) == {'dispatches_per_year_6_day','dispatches_per_year_12_day','days_per_dispatch_6_day','days_per_dispatch_12_day','switch_threshold_usd','override_rule','suspension_rule'}
for c in data['campaigns']:
    expected_keys = {'campaign_id','campaign_name','active_clinics','drug_cost_per_1000_doses_usd','doses_per_day','crate_tier','crate_cost_usd','payment_per_dispatch_per_clinic_usd','annual_drug_cost_6_day_usd','annual_drug_cost_12_day_usd','annual_crate_cost_6_day_usd','annual_crate_cost_12_day_usd','annual_revenue_6_day_usd','annual_revenue_12_day_usd','annual_margin_6_day_usd','annual_margin_12_day_usd','annual_margin_difference_12_minus_6_usd'}
    assert set(c.keys()) == expected_keys, f'Bad keys: {set(c.keys()) - expected_keys} extra, {expected_keys - set(c.keys())} missing'
assert data['campaigns'] == sorted(data['campaigns'], key=lambda x: x['campaign_id'])
print('JSON schema OK')
print('Totals:', json.dumps(data['totals'], indent=2))
print('Decision:', data['recommendation']['decision'])
"
```

```bash
cat /root/vaxcrate_summary.md
```

Verify the summary contains the comma-formatted totals and the decision slug.

## Step 4 – Debug if needed
If any validation fails, re-read the input files, trace the calculation for one campaign by hand, fix the script, and re-run. Pay special attention to:
- Whether `crate_cost_usd` is per-dispatch or annual (check the CSV values and column names)
- Whether `alias_labels` in the manifest is a list or string
- Proper exclusion of suspended campaigns
- Proper filtering of location overrides (approved, non-blank revision and active_clinics)
- Keeping the latest cycle_tag among active billing rows per campaign

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