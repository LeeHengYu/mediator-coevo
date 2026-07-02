# Task Instruction

Execute the following steps in order. Read every input file before writing any code.

## Step 1 — Inspect all input files

```bash
cat /root/campaign_manifest.json
cat /root/crate_cost.csv
cat /root/billing.csv
cat /root/location_overrides.csv
cat /root/suspensions.csv
```

## Step 2 — Write and run a Python script that produces both output files

Create `/root/solve.py` with the logic below, then run it with `python3 /root/solve.py`.

### Logic the script must implement (in order):

1. **Load inputs.**
   - `campaign_manifest.json` → list of campaign objects.
   - `crate_cost.csv`, `billing.csv`, `location_overrides.csv`, `suspensions.csv` → parse as CSV (use `csv.DictReader`).

2. **Filter campaigns.**
   - Keep only campaigns where `analysis_flag == "review"`.
   - From `suspensions.csv`, collect every `campaign_id` whose `suspension_status` is `"hold"`. Remove any campaign whose `campaign_id` is in that set.

3. **Resolve billing rows.**
   - For each retained campaign, find rows in `billing.csv` where `campaign_label` matches either `campaign_name` OR any element of `alias_labels` (which is a list in the manifest).
   - Keep only rows with `status == "active"`.
   - If multiple active rows map to the same campaign, keep the one with the latest (lexicographically greatest) `cycle_tag`.
   - Extract `payment_per_dispatch_per_clinic_usd` (convert to float).

4. **Resolve active clinics from location_overrides.csv.**
   - Keep rows where `state == "approved"`.
   - Discard rows where `revision` is blank/empty or `active_clinics` is blank/empty.
   - Among remaining rows for the same `campaign_id`, keep the one with the highest numeric `revision`.
   - If no valid override row exists for a campaign, use `default_active_clinics` from the manifest.
   - `active_clinics` must be an integer.

5. **Look up crate cost.**
   - Each campaign has a `crate_tier` in the manifest. Match it against `crate_cost.csv` to get `crate_cost_usd` (float).

6. **Compute per-campaign numbers (all floats, rounded to 2 decimals at output).**

   Constants:
   - 6-day: `days_per_dispatch=6`, `dispatches_per_year=60`
   - 12-day: `days_per_dispatch=12`, `dispatches_per_year=30`

   For each model (6-day and 12-day):
   - `annual_revenue = payment_per_dispatch_per_clinic_usd * active_clinics * dispatches_per_year`
   - `annual_drug_cost = drug_cost_per_1000_doses_usd * active_clinics * doses_per_day * days_per_dispatch * dispatches_per_year / 1000`
   - `annual_crate_cost = crate_cost_usd * dispatches_per_year`  (NOTE: the task says "annual_crate_cost" — crate cost is per dispatch, so multiply by dispatches_per_year. Inspect the data to confirm this interpretation makes sense. If the crate_cost.csv already looks like an annual figure or the numbers don't make sense, re-read the task.)
   - `annual_margin = annual_revenue - annual_drug_cost - annual_crate_cost`
   - `difference = annual_margin_12_day - annual_margin_6_day`

7. **Totals.**
   - Sum all per-campaign margins for each model.
   - `total_difference = sum of per-campaign differences` (equivalently total_12 − total_6).
   - `absolute_total_margin_difference = abs(total_difference)`.

8. **Decision.**
   - If `abs(total_difference) < 11000` → `"move_to_12_day"`.
   - Otherwise → `"keep_6_day"`.

9. **Build JSON output** exactly matching the schema in the task. Round every currency value to 2 decimal places. Sort `campaigns` array by `campaign_id` ascending. Write to `/root/vaxcrate_analysis.json` with `json.dump(..., indent=2)`.

10. **Build markdown summary** `/root/vaxcrate_summary.md` with 4–8 non-empty lines containing:
    - Total 6-day margin (USD)
    - Total 12-day margin (USD)
    - Absolute difference (USD)
    - The exact decision slug (`move_to_12_day` or `keep_6_day`)

## Step 3 — Validate outputs

```bash
python3 -c "
import json, sys
with open('/root/vaxcrate_analysis.json') as f:
    d = json.load(f)
assert 'assumptions' in d
assert 'campaigns' in d and len(d['campaigns']) > 0
assert 'totals' in d
assert 'recommendation' in d
ids = [c['campaign_id'] for c in d['campaigns']]
assert ids == sorted(ids), 'campaigns not sorted by campaign_id'
for c in d['campaigns']:
    for k in ['annual_drug_cost_6_day_usd','annual_drug_cost_12_day_usd','annual_crate_cost_6_day_usd','annual_crate_cost_12_day_usd','annual_revenue_6_day_usd','annual_revenue_12_day_usd','annual_margin_6_day_usd','annual_margin_12_day_usd','annual_margin_difference_12_minus_6_usd']:
        assert k in c, f'Missing key {k}'
print('JSON structure OK')

with open('/root/vaxcrate_summary.md') as f:
    lines = [l for l in f.read().strip().splitlines() if l.strip()]
assert 4 <= len(lines) <= 8, f'Expected 4-8 non-empty lines, got {len(lines)}'
dec = d['recommendation']['decision']
assert dec in ('move_to_12_day','keep_6_day')
md_text = open('/root/vaxcrate_summary.md').read()
assert dec in md_text, 'Decision slug not found in summary'
print('Summary OK')
print('All checks passed')
"
```

If any check fails, read the error, fix the script, and re-run until both output files pass validation.

## Important edge-case reminders
- `alias_labels` in the manifest is a list; iterate through it when matching billing rows.
- `cycle_tag` comparison: treat as string and pick lexicographic max (works for date-like tags like "2024-Q3" etc.).
- `revision` in location_overrides: convert to int/float for numeric comparison; skip blanks.
- Crate cost: the formula `annual_crate_cost = crate_cost_usd * dispatches_per_year` is the natural reading. If the numbers look implausible, double-check by inspecting the CSV values and re-reading the task.
- All monetary values rounded to exactly 2 decimal places in the output JSON.
- The `justification` string should briefly explain the decision referencing the threshold and the absolute difference.

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