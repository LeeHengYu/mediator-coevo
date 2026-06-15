# Task Instruction

Execute the following steps in a single Python script to produce `/root/infusion_batch_analysis.json` and `/root/infusion_batch_summary.md`.

### Step 0 – Inspect input files
Before writing any logic, read and print the first few lines/entries of each input file so you understand their exact structure:
- `/root/therapy_catalog.json`
- `/root/bag_supply_cost.csv`
- `/root/delivery_payment.csv`
- `/root/patient_overrides.csv`

### Step 1 – Load data
```python
import json, csv, pathlib

with open('/root/therapy_catalog.json') as f:
    catalog = json.load(f)

def read_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))

bag_rows = read_csv('/root/bag_supply_cost.csv')
payment_rows = read_csv('/root/delivery_payment.csv')
override_rows = read_csv('/root/patient_overrides.csv')
```

### Step 2 – Filter in-scope therapies
From `therapy_catalog.json`, keep only entries where `include_in_review` is `true` (boolean). Build a dict keyed by `therapy_code`.

### Step 3 – Build alias → therapy_code lookup
For each in-scope therapy, map its `therapy_name` AND every entry in its `aliases` list (if present) to its `therapy_code`. Use this to resolve `therapy_label` in `delivery_payment.csv`.

### Step 4 – Resolve delivery payment
For each row in `delivery_payment.csv`, look up `therapy_label` in the alias map. Skip rows that don't match any in-scope therapy. Store `payment_per_delivery_per_patient_usd` (float) keyed by `therapy_code`.

### Step 5 – Resolve bag supply cost
Build a dict from `bag_supply_cost.csv`: key = `bag_size_ml` (int), value = `bag_supply_cost_usd` (float).

### Step 6 – Resolve active patients from overrides
Filter `patient_overrides.csv` to rows where `status` == `approved`. Among those, for each `therapy_code`, keep only the row with the highest `revision` (int). Discard rows whose `therapy_code` is not in scope. Store `active_patients` (int) keyed by `therapy_code`.

### Step 7 – Compute per-therapy metrics
For each in-scope therapy (sorted by `therapy_code` ascending):

```
drug_cost_per_1000_mg = catalog entry's drug_cost_per_1000_mg_usd
dose_mg_per_day = catalog entry's dose_mg_per_day
bag_size_ml = catalog entry's bag_size_ml
bag_supply_cost = looked up from bag cost dict
payment = from delivery payment dict
patients = from overrides dict

For model in [7-day, 14-day]:
  days_per_delivery = 7 or 14
  deliveries_per_year = 52 or 26

  annual_drug_cost = drug_cost_per_1000_mg * patients * dose_mg_per_day * days_per_delivery * deliveries_per_year / 1000
  annual_supply_cost = bag_supply_cost * patients * deliveries_per_year
  annual_revenue = payment * patients * deliveries_per_year
  annual_margin = annual_revenue - annual_drug_cost - annual_supply_cost

margin_diff = annual_margin_14_day - annual_margin_7_day
```

Round ALL currency values to 2 decimal places using `round(value, 2)`.

### Step 8 – Compute totals
```
total_margin_7 = sum of all annual_margin_7_day
total_margin_14 = sum of all annual_margin_14_day
total_diff = total_margin_14 - total_margin_7   (also equals sum of per-therapy diffs)
abs_diff = abs(total_diff)
```
Round each to 2 decimals.

### Step 9 – Decision
```
if abs_diff < 15000:
    decision = "move_to_14_day"
else:
    decision = "keep_7_day"
```

Provide a short justification string.

### Step 10 – Write JSON output
Write `/root/infusion_batch_analysis.json` with the exact schema from the task. Use `json.dump` with `indent=2`. All numeric values must be plain numbers (no commas, no strings).

### Step 11 – Write Markdown summary
Write `/root/infusion_batch_summary.md` with 4–8 non-empty lines. **Critical formatting requirement for the markdown file:**

You must determine the correct number formatting by inspecting the test file. Before writing the markdown, do this:

```python
import glob, os
test_files = glob.glob('/root/**/test*.py', recursive=True)
for tf in test_files:
    with open(tf) as f:
        content = f.read()
    print(f'=== {tf} ===')
    print(content)
```

Look at the test assertions for the summary file. If the test checks for comma-formatted numbers (like `'-455,619.31'`), use `f"{value:,.2f}"`. If it checks for plain numbers (like `'-455619.31'`), use `f"{value:.2f}"`. Match whatever the test expects.

The markdown must include:
- Total 7-day margin (USD)
- Total 14-day margin (USD)
- Absolute difference (USD)
- Final decision using the exact slug `move_to_14_day` or `keep_7_day`

### Step 12 – Verify outputs
After writing both files:
1. Re-read and print `/root/infusion_batch_analysis.json` to confirm valid JSON and correct schema.
2. Re-read and print `/root/infusion_batch_summary.md` to confirm formatting.
3. If test files were found, run them with `python -m pytest <test_file> -v` and report results.

### Important notes
- The `annual_supply_cost` formula is: `bag_supply_cost_usd * active_patients * deliveries_per_year`. Note that drug cost uses `days_per_delivery` in its formula but supply cost does NOT — supply cost is per-delivery (one bag set per delivery).
- Be careful with field name lookups — inspect the actual JSON/CSV headers before assuming names.
- If a therapy has no matching payment row or no approved patient override row, investigate whether it should still appear (it likely should with 0 patients or needs special handling — check the data).

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
Task metadata: author_email=gpt54@example.com, author_name=GPT-5.4, category=financial-analysis, difficulty=medium, tags=[home-infusion, json, csv, alias-resolution, decision-analysis].
Verifier config: timeout_sec=900.0.