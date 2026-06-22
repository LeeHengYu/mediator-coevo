# Task Instruction

Execute the following steps in order to produce `/root/vaxcrate_analysis.json` and `/root/vaxcrate_summary.md`.

## Step 0 – Inspect all input files

```bash
cat /root/campaign_manifest.json
cat /root/crate_cost.csv
cat /root/billing.csv
cat /root/location_overrides.csv
cat /root/suspensions.csv
```

Read every file carefully before writing any code.

## Step 1 – Write and run `solve.py`

Create `/root/solve.py` with the logic below. Follow every rule exactly.

### 1. Load data
- `campaign_manifest.json` → list of campaign objects.
- `crate_cost.csv`, `billing.csv`, `location_overrides.csv`, `suspensions.csv` → CSVs (use `csv.DictReader`).

### 2. Filter campaigns
- Keep only campaigns where `analysis_flag == "review"`.
- Build a set of suspended campaign_ids: rows in `suspensions.csv` where `suspension_status == "hold"`.
- Remove any campaign whose `campaign_id` is in that set.

### 3. Resolve billing
- For each retained campaign, find matching rows in `billing.csv` where `campaign_label` equals either `campaign_name` OR any element in the campaign's `alias_labels` list.
- Keep only rows where `status == "active"`.
- If multiple active rows match the same campaign, keep the one with the lexicographically latest `cycle_tag`.
- Extract `payment_per_dispatch_per_clinic_usd` (float) from the kept row.

### 4. Resolve active clinics
- From `location_overrides.csv`, keep rows where `state == "approved"` AND `revision` is not blank AND `active_clinics` is not blank.
- Group valid rows by `campaign_id`. For each group, keep the row with the highest numeric `revision`.
- For each retained campaign: if an override row exists, use its `active_clinics` (int). Otherwise use the campaign's `default_active_clinics` (int) from the manifest.

### 5. Resolve crate cost
- Build a dict from `crate_cost.csv`: `crate_tier` → `crate_cost_usd` (float).
- For each campaign, look up `crate_cost_usd` by the campaign's `crate_tier`.

### 6. Compute per-campaign numbers
Constants:
- 6-day: `days_per_dispatch=6`, `dispatches_per_year=60`
- 12-day: `days_per_dispatch=12`, `dispatches_per_year=30`

For each model (6 and 12):
- `annual_revenue = payment_per_dispatch_per_clinic_usd * active_clinics * dispatches_per_year`
- `annual_drug_cost = drug_cost_per_1000_doses_usd * active_clinics * doses_per_day * days_per_dispatch * dispatches_per_year / 1000`
- `annual_crate_cost = crate_cost_usd * active_clinics * dispatches_per_year`  (This follows the pattern confirmed in harbor_oncocooler_10v20: cost * sites * dispatches.)
- `annual_margin = annual_revenue - annual_drug_cost - annual_crate_cost`
- `difference = annual_margin_12 - annual_margin_6`

Round every currency value to 2 decimal places.

### 7. Build output JSON
The JSON must match the schema EXACTLY:
```
{
  "assumptions": { ... },   // fixed literal object as given in prompt
  "campaigns": [ ... ],      // sorted by campaign_id ascending
  "totals": {
    "total_annual_margin_6_day_usd": ...,
    "total_annual_margin_12_day_usd": ...,
    "total_annual_margin_difference_12_minus_6_usd": ...,
    "absolute_total_margin_difference_usd": ...
  },
  "recommendation": {
    "decision": "move_to_12_day" or "keep_6_day",
    "justification": "<string>"
  }
}
```

Each campaign object MUST include ALL of these keys (no more, no fewer):
`campaign_id`, `campaign_name`, `active_clinics`, `drug_cost_per_1000_doses_usd`, `doses_per_day`, `crate_tier`, `crate_cost_usd`, `payment_per_dispatch_per_clinic_usd`, `annual_drug_cost_6_day_usd`, `annual_drug_cost_12_day_usd`, `annual_crate_cost_6_day_usd`, `annual_crate_cost_12_day_usd`, `annual_revenue_6_day_usd`, `annual_revenue_12_day_usd`, `annual_margin_6_day_usd`, `annual_margin_12_day_usd`, `annual_margin_difference_12_minus_6_usd`.

Decision rule: if `abs(total_difference) < 11000` → `move_to_12_day`, else `keep_6_day`.

`recommendation.decision` must be the exact slug. `recommendation.justification` should be a short sentence referencing the absolute difference and the threshold.

### 8. Build summary markdown `/root/vaxcrate_summary.md`
4–8 non-empty lines. Must include:
- Total 6-day margin with commas and 2 decimals, e.g. `$-83,406.84`
- Total 12-day margin similarly formatted
- Absolute difference similarly formatted
- The exact decision slug (`move_to_12_day` or `keep_6_day`)

Use `f"{value:,.2f}"` for formatting currency in the markdown.

### 9. Write files
Write JSON with `json.dump(..., indent=2)`. Write markdown as plain text.

## Step 2 – Run
```bash
python3 /root/solve.py
```

## Step 3 – Validate
```bash
python3 -c "
import json
data = json.load(open('/root/vaxcrate_analysis.json'))
assert 'assumptions' in data
assert 'campaigns' in data
assert 'totals' in data
assert 'recommendation' in data
assert 'decision' in data['recommendation']
assert 'justification' in data['recommendation']
for c in data['campaigns']:
    for k in ['campaign_id','campaign_name','active_clinics','drug_cost_per_1000_doses_usd','doses_per_day','crate_tier','crate_cost_usd','payment_per_dispatch_per_clinic_usd','annual_drug_cost_6_day_usd','annual_drug_cost_12_day_usd','annual_crate_cost_6_day_usd','annual_crate_cost_12_day_usd','annual_revenue_6_day_usd','annual_revenue_12_day_usd','annual_margin_6_day_usd','annual_margin_12_day_usd','annual_margin_difference_12_minus_6_usd']:
        assert k in c, f'Missing key {k} in campaign {c.get(\"campaign_id\",\"?\")}'  
ids = [c['campaign_id'] for c in data['campaigns']]
assert ids == sorted(ids), 'campaigns not sorted by campaign_id'
print('JSON schema OK')
print('Totals:', json.dumps(data['totals'], indent=2))
print('Decision:', data['recommendation']['decision'])
"
cat /root/vaxcrate_summary.md
```

If any assertion fails or the numbers look wrong, debug and fix before finishing.

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