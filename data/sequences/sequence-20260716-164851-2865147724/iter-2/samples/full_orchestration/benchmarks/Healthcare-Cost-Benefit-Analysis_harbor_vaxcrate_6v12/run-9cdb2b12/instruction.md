# Task Instruction

Execute the following steps in order to produce `/root/vaxcrate_analysis.json` and `/root/vaxcrate_summary.md`.

## Step 1 – Inspect all input files

Read and display the full contents of:
- `/root/campaign_manifest.json`
- `/root/crate_cost.csv`
- `/root/billing.csv`
- `/root/location_overrides.csv`
- `/root/suspensions.csv`

Understand every column name and data type before proceeding.

## Step 2 – Write and run a Python script

Create `/root/solve.py` that does **all** of the following, then run it with `python3 /root/solve.py`.

### 2a – Load data
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

### 2b – Filter campaigns
1. From `manifest`, keep only entries where `analysis_flag == "review"`.
2. Build a set of suspended campaign_ids: those appearing in `suspensions.csv` with `suspension_status == "hold"`.
3. Exclude any campaign whose `campaign_id` is in that suspended set.

### 2c – Resolve billing rows
For each retained campaign:
1. Match `billing.csv` rows where `campaign_label` equals `campaign_name` **or** is found in the campaign's `alias_labels` list.
2. Keep only rows with `status == "active"`.
3. If multiple active rows match, keep the one with the latest (lexicographically largest) `cycle_tag`.
4. Extract `payment_per_dispatch_per_clinic_usd` (convert to float).

### 2d – Resolve active clinics from location_overrides
For each retained campaign:
1. Filter `location_overrides.csv` to rows matching `campaign_id`, `state == "approved"`, non-blank `revision`, and non-blank `active_clinics`.
2. Among those, pick the row with the highest numeric `revision`.
3. Use its `active_clinics` (convert to int).
4. If no qualifying row exists, use `default_active_clinics` from the manifest entry.

### 2e – Look up crate cost
For each campaign, use `crate_tier` from the manifest to look up `crate_cost_usd` in `crate_cost.csv` (convert to float).

### 2f – Compute per-campaign figures
Constants:
- 6-day: days_per_dispatch=6, dispatches_per_year=60
- 12-day: days_per_dispatch=12, dispatches_per_year=30

For each campaign, using values from the manifest (`drug_cost_per_1000_doses_usd`, `doses_per_day`) and the resolved `active_clinics`, `crate_cost_usd`, `payment_per_dispatch_per_clinic_usd`:

```
annual_revenue = payment_per_dispatch_per_clinic_usd * active_clinics * dispatches_per_year
annual_drug_cost = drug_cost_per_1000_doses_usd * active_clinics * doses_per_day * days_per_dispatch * dispatches_per_year / 1000
annual_crate_cost = crate_cost_usd * dispatches_per_year   # (crate_cost_usd is per dispatch)
annual_margin = annual_revenue - annual_drug_cost - annual_crate_cost
```

**Important**: Note the crate cost formula. The task says "annual_crate_cost" uses `crate_cost_usd` from `crate_cost.csv`. Read the `crate_cost.csv` carefully – if there is a `per_dispatch` or similar qualifier, multiply by dispatches_per_year. If the CSV value already represents an annual figure, use it directly. Inspect the CSV header and values to decide. The most natural reading is: `annual_crate_cost = crate_cost_usd * dispatches_per_year`.

Compute for both 6-day and 12-day models.

```
difference = annual_margin_12_day - annual_margin_6_day
```

Round every currency value to 2 decimal places.

### 2g – Totals and decision
```
total_margin_6 = sum of all annual_margin_6_day
total_margin_12 = sum of all annual_margin_12_day
total_diff = total_margin_12 - total_margin_6
abs_diff = abs(total_diff)
```
Round to 2 decimals.

Decision rule:
- If `abs_diff < 11000` → `move_to_12_day`
- Otherwise → `keep_6_day`

### 2h – Build and write JSON output
Sort the campaigns list by `campaign_id` ascending.

Write `/root/vaxcrate_analysis.json` with the exact schema from the task, using `json.dump` with `indent=2`.

Include the `assumptions` block exactly as specified (with the literal strings for `override_rule` and `suspension_rule`).

For `recommendation.justification`, write a short sentence referencing the absolute difference and threshold.

### 2i – Write summary markdown
Write `/root/vaxcrate_summary.md` with 4-8 non-empty lines including:
- Total 6-day margin in USD
- Total 12-day margin in USD
- Absolute difference in USD
- The exact decision slug (`move_to_12_day` or `keep_6_day`)

## Step 3 – Validate

1. `cat /root/vaxcrate_analysis.json` and verify it parses as valid JSON, has the correct schema, campaigns are sorted by campaign_id, and all currency values have at most 2 decimal places.
2. `cat /root/vaxcrate_summary.md` and verify it has 4-8 non-empty lines and contains the required information.
3. Run `python3 -c "import json; d=json.load(open('/root/vaxcrate_analysis.json')); print('campaigns:', len(d['campaigns'])); print('decision:', d['recommendation']['decision'])"` to confirm the JSON is well-formed.

If any step fails, diagnose and fix before proceeding to the next step.

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