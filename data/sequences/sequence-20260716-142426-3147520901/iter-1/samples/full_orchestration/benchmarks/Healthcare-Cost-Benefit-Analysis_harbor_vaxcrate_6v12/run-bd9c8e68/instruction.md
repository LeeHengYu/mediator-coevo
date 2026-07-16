# Task Instruction

Execute the following steps in order.

## 1. Inspect all input files

```bash
cat /root/campaign_manifest.json
cat /root/crate_cost.csv
cat /root/billing.csv
cat /root/location_overrides.csv
cat /root/suspensions.csv
```

## 2. Write and run a Python script to produce both output files

Create `/root/solve.py` with the following logic, then run it with `python3 /root/solve.py`.

### Logic the script must implement

**A. Load data**
- `campaign_manifest.json` – list/dict of campaigns. Each has at minimum: `campaign_id`, `campaign_name`, `alias_labels` (list of strings), `analysis_flag`, `default_active_clinics`, `drug_cost_per_1000_doses_usd`, `doses_per_day`, `crate_tier`.
- `crate_cost.csv` – columns include `crate_tier`, `crate_cost_usd`.
- `billing.csv` – columns include `campaign_label`, `status`, `cycle_tag`, `payment_per_dispatch_per_clinic_usd`.
- `location_overrides.csv` – columns include `campaign_id`, `state`, `revision`, `active_clinics`.
- `suspensions.csv` – columns include `campaign_id`, `suspension_status`.

**B. Filter campaigns**
1. From manifest, keep only campaigns where `analysis_flag == "review"`.
2. Exclude any campaign whose `campaign_id` appears in `suspensions.csv` with `suspension_status == "hold"`.

**C. Resolve billing**
For each retained campaign:
1. Find billing rows where `campaign_label` matches `campaign_name` OR any element of `alias_labels`.
2. Keep only rows with `status == "active"`.
3. If multiple, keep the row with the latest (lexicographically largest) `cycle_tag`.
4. Extract `payment_per_dispatch_per_clinic_usd` from that row.

**D. Resolve active clinics**
For each retained campaign:
1. From `location_overrides.csv`, filter rows matching `campaign_id`, `state == "approved"`, `revision` is not blank/empty, `active_clinics` is not blank/empty.
2. Among those, keep the row with the highest numeric `revision`.
3. Use its `active_clinics` (as a number).
4. If no valid row exists, use `default_active_clinics` from manifest.

**E. Look up crate cost**
For each retained campaign, match `crate_tier` from manifest to `crate_cost.csv` to get `crate_cost_usd`.

**F. Compute per-campaign figures (round each to 2 decimals)**
- 6-day model: days_per_dispatch=6, dispatches_per_year=60
- 12-day model: days_per_dispatch=12, dispatches_per_year=30

For each model:
- `annual_revenue = payment_per_dispatch_per_clinic_usd * active_clinics * dispatches_per_year`
- `annual_drug_cost = drug_cost_per_1000_doses_usd * active_clinics * doses_per_day * days_per_dispatch * dispatches_per_year / 1000`
- `annual_crate_cost = crate_cost_usd * dispatches_per_year`  (crate cost is per dispatch, so multiply by dispatches_per_year)
  **IMPORTANT**: Re-read the input files carefully. If `crate_cost.csv` has a column like `crate_cost_usd` that already represents per-crate or per-dispatch cost, use it directly times dispatches_per_year. If the manifest or data suggests a different interpretation, inspect and adapt. The key formula is: `annual_crate_cost = crate_cost_usd * dispatches_per_year`. Double-check by inspecting the crate_cost.csv header and values.
- `annual_margin = annual_revenue - annual_drug_cost - annual_crate_cost`
- `difference = annual_margin_12_day - annual_margin_6_day`

Round every currency value to 2 decimal places.

**G. Totals**
- Sum all per-campaign `annual_margin_6_day_usd` → `total_annual_margin_6_day_usd`
- Sum all per-campaign `annual_margin_12_day_usd` → `total_annual_margin_12_day_usd`
- `total_annual_margin_difference_12_minus_6_usd = total_12 - total_6`
- `absolute_total_margin_difference_usd = abs(total_difference)`
- Round all to 2 decimals.

**H. Decision**
- If `abs(total_difference) < 11000` → `"move_to_12_day"`
- Otherwise → `"keep_6_day"`
- Justification: a brief string explaining the numbers.

**I. Sort campaigns by `campaign_id` ascending.**

**J. Write `/root/vaxcrate_analysis.json`** with the exact schema from the task, using `json.dump` with `indent=2`.

**K. Write `/root/vaxcrate_summary.md`** with 4–8 non-empty lines containing:
- Total 6-day margin (USD)
- Total 12-day margin (USD)
- Absolute difference (USD)
- Final decision slug (`move_to_12_day` or `keep_6_day`)

## 3. Validate outputs

```bash
cat /root/vaxcrate_analysis.json
cat /root/vaxcrate_summary.md
python3 -c "
import json
with open('/root/vaxcrate_analysis.json') as f:
    d = json.load(f)
assert 'assumptions' in d
assert 'campaigns' in d and len(d['campaigns']) > 0
assert 'totals' in d
assert 'recommendation' in d
assert d['recommendation']['decision'] in ('move_to_12_day', 'keep_6_day')
ids = [c['campaign_id'] for c in d['campaigns']]
assert ids == sorted(ids), 'campaigns not sorted by campaign_id'
print('JSON validation passed')
with open('/root/vaxcrate_summary.md') as f:
    lines = [l for l in f.read().strip().split('\\n') if l.strip()]
assert 4 <= len(lines) <= 8, f'Expected 4-8 non-empty lines, got {len(lines)}'
print('Summary validation passed')
"
```

## Important notes
- Read every input file carefully before writing the script. Field names may differ slightly from what's described (e.g., extra whitespace, different casing). Adapt accordingly.
- Pay special attention to `alias_labels` — it may be a JSON list stored as a string or a proper list. Parse it correctly.
- For `cycle_tag` comparison, treat as string and use lexicographic ordering.
- For `revision`, convert to numeric (int or float) for comparison.
- Handle edge cases: empty strings, missing fields, type mismatches.
- Do NOT skip or invent data. Use only what's in the files.
- If `crate_cost.csv` has additional columns or different structure, adapt. Inspect first.

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