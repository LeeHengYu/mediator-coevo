# Task Instruction

Execute the following steps in order:

## 1. Inspect all input files

Read and display the contents of:
- `/root/assay_manifest.json`
- `/root/carrier_cost.csv`
- `/root/billing.csv`
- `/root/lab_overrides.csv`
- `/root/report_template.json`

## 2. Write a Python script `/root/solve.py` that does the following:

```python
import json
import csv
from collections import defaultdict

# Load assay_manifest.json
with open('/root/assay_manifest.json') as f:
    manifest = json.load(f)

# Load carrier_cost.csv
carrier_costs = {}
with open('/root/carrier_cost.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        carrier_costs[row['carrier_type'].strip()] = float(row['carrier_cost_usd'])

# Load billing.csv
billing_rows = []
with open('/root/billing.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        billing_rows.append(row)

# Load lab_overrides.csv
override_rows = []
with open('/root/lab_overrides.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        override_rows.append(row)

# Load report_template.json
with open('/root/report_template.json') as f:
    template = json.load(f)

# Step A: Filter in-scope assays
in_scope_assays = [a for a in manifest['assays'] if a.get('in_scope') == True]

# Step B: Build alias->assay_id mapping
# For each in-scope assay, map assay_name and all aliases to that assay
assay_label_to_assay = {}
for a in in_scope_assays:
    assay_label_to_assay[a['assay_name'].strip()] = a['assay_id']
    for alias in a.get('aliases', []):
        assay_label_to_assay[alias.strip()] = a['assay_id']

# Step C: Resolve billing rows
# Filter active billing rows, match to assay_id, keep latest effective_month per assay
active_billing = [b for b in billing_rows if b.get('is_active', '').strip().lower() == 'true']

billing_by_assay = {}
for b in active_billing:
    label = b['assay_label'].strip()
    if label in assay_label_to_assay:
        aid = assay_label_to_assay[label]
        em = b['effective_month'].strip()
        if aid not in billing_by_assay or em > billing_by_assay[aid]['effective_month'].strip():
            billing_by_assay[aid] = b

# Step D: Resolve active labs from lab_overrides
approved_overrides = [r for r in override_rows if r.get('status', '').strip().lower() == 'approved']
override_by_assay = {}
for r in approved_overrides:
    aid = r['assay_id'].strip()
    rev = int(r['revision'])
    if aid not in override_by_assay or rev > override_by_assay[aid][1]:
        override_by_assay[aid] = (int(r['active_labs']), rev)

# Step E: Build per-assay analysis
assay_results = []
for a in in_scope_assays:
    aid = a['assay_id']
    aname = a['assay_name']
    
    # Active labs
    if aid in override_by_assay:
        active_labs = override_by_assay[aid][0]
    else:
        active_labs = int(a['default_active_labs'])
    
    reagent_price = float(a['reagent_price_per_1000_tests_usd'])
    carrier_type = a['carrier_type'].strip()
    carrier_cost = carrier_costs[carrier_type]
    
    billing_row = billing_by_assay[aid]
    payment_per_run = float(billing_row['payment_per_run_per_lab_usd'])
    
    tests_small = int(a['tests_per_lab_per_run_small'])
    tests_bulk = int(a['tests_per_lab_per_run_bulk'])
    
    runs_small = 24
    runs_bulk = 12
    
    # Annual revenue
    rev_small = payment_per_run * active_labs * runs_small
    rev_bulk = payment_per_run * active_labs * runs_bulk
    
    # Annual reagent cost
    reagent_small = reagent_price * active_labs * tests_small * runs_small / 1000
    reagent_bulk = reagent_price * active_labs * tests_bulk * runs_bulk / 1000
    
    # Annual carrier cost - carrier_cost per shipment, runs_per_year shipments
    # The carrier_cost_usd is per shipment. Each run requires a shipment.
    carrier_small = carrier_cost * active_labs * runs_small
    carrier_bulk = carrier_cost * active_labs * runs_bulk
    
    # Wait - need to check what annual_carrier_cost means. The formula says:
    # annual_margin = annual_revenue - annual_reagent_cost - annual_carrier_cost
    # carrier_cost_usd from carrier_cost.csv matched by carrier_type
    # It's likely carrier_cost * runs_per_year (one shipment per run per lab? or total?)
    # Let me re-check: the schema has carrier_cost_usd as a single value.
    # The annual_carrier_cost is probably carrier_cost_usd * runs_per_year
    # But we need to check if it's per lab or total. Given the revenue formula
    # includes active_labs, and reagent cost includes active_labs, carrier likely does too.
    # Actually let me just keep carrier_small = carrier_cost * runs_small (no per-lab)
    # Hmm, but that seems inconsistent. Let me check the schema fields...
    # The schema just has annual_carrier_cost_small_kit_usd and annual_carrier_cost_bulk_kit_usd
    # Without explicit formula given, I'll assume carrier_cost * runs_per_year (not per lab)
    # since carrier might be a flat per-shipment cost to the lab network.
    # Actually, re-reading: no explicit carrier cost formula is given. Let me try
    # carrier_cost_usd * runs_per_year (without active_labs multiplier) first.
    # If that doesn't work, I'll try with active_labs.
    # Actually, thinking about it more carefully: each lab needs reagents shipped.
    # So it should be carrier_cost * active_labs * runs_per_year.
    # Let me keep that.
    
    margin_small = rev_small - reagent_small - carrier_small
    margin_bulk = rev_bulk - reagent_bulk - carrier_bulk
    diff = margin_bulk - margin_small
    
    assay_results.append({
        'assay_id': aid,
        'assay_name': aname,
        'active_labs': active_labs,
        'reagent_price_per_1000_tests_usd': round(reagent_price, 2),
        'carrier_type': carrier_type,
        'carrier_cost_usd': round(carrier_cost, 2),
        'payment_per_run_per_lab_usd': round(payment_per_run, 2),
        'tests_per_lab_per_run_small': tests_small,
        'tests_per_lab_per_run_bulk': tests_bulk,
        'annual_reagent_cost_small_kit_usd': round(reagent_small, 2),
        'annual_reagent_cost_bulk_kit_usd': round(reagent_bulk, 2),
        'annual_carrier_cost_small_kit_usd': round(carrier_small, 2),
        'annual_carrier_cost_bulk_kit_usd': round(carrier_bulk, 2),
        'annual_revenue_small_kit_usd': round(rev_small, 2),
        'annual_revenue_bulk_kit_usd': round(rev_bulk, 2),
        'annual_margin_small_kit_usd': round(margin_small, 2),
        'annual_margin_bulk_kit_usd': round(margin_bulk, 2),
        'annual_margin_difference_bulk_minus_small_usd': round(diff, 2)
    })

# Sort by assay_id ascending
assay_results.sort(key=lambda x: x['assay_id'])

# Totals
total_margin_small = sum(a['annual_margin_small_kit_usd'] for a in assay_results)
total_margin_bulk = sum(a['annual_margin_bulk_kit_usd'] for a in assay_results)
total_diff = round(total_margin_bulk - total_margin_small, 2)
abs_diff = round(abs(total_diff), 2)

# Decision
if abs_diff < 7000:
    decision = 'adopt_bulk_kit'
    justification = f'Absolute margin difference ${abs_diff} is below $7000 threshold; bulk-kit policy is acceptable.'
else:
    decision = 'keep_small_kit'
    justification = f'Absolute margin difference ${abs_diff} exceeds $7000 threshold; keeping small-kit policy.'

# Build report
report = {
    'metadata': template['metadata'],
    'analysis': {
        'assumptions': {
            'runs_per_year_small_kit': 24,
            'runs_per_year_bulk_kit': 12,
            'switch_threshold_usd': 7000,
            'lab_override_rule': 'highest approved revision per assay_id, else default_active_labs',
            'billing_rule': 'latest active effective_month per assay'
        },
        'assays': assay_results,
        'totals': {
            'total_annual_margin_small_kit_usd': round(total_margin_small, 2),
            'total_annual_margin_bulk_kit_usd': round(total_margin_bulk, 2),
            'total_annual_margin_difference_bulk_minus_small_usd': total_diff,
            'absolute_total_margin_difference_usd': abs_diff
        },
        'recommendation': {
            'decision': decision,
            'justification': justification
        }
    }
}

with open('/root/reagent_policy_report.json', 'w') as f:
    json.dump(report, f, indent=2)

# Write summary markdown
lines = [
    '# Reagent Policy Summary',
    f'Total small-kit annual margin: ${round(total_margin_small, 2)}',
    f'Total bulk-kit annual margin: ${round(total_margin_bulk, 2)}',
    f'Absolute margin difference: ${abs_diff}',
    f'Decision: {decision}',
    justification
]

with open('/root/reagent_policy_summary.md', 'w') as f:
    f.write('\n'.join(lines) + '\n')

print('Done. Report and summary written.')
print(json.dumps(report, indent=2))
```

**IMPORTANT**: Before writing the script, first inspect all input files carefully. The script above is a template — you MUST adapt it based on the actual column names, field names, and data structures found in the input files. Pay special attention to:
- Exact field names in `assay_manifest.json` (e.g., `in_scope`, `aliases`, `default_active_labs`, `carrier_type`, etc.)
- Exact column headers in CSV files (strip whitespace)
- Whether `is_active` in billing.csv is a string 'true'/'True' or boolean
- Whether `in_scope` in the manifest is a boolean or string
- The structure of `report_template.json` metadata

**CRITICAL SCHEMA NOTE** (from cross-task failure feedback): The output JSON must use FLAT keys exactly matching the schema. Do NOT nest small-kit/bulk-kit values into sub-objects. Every key in the schema (e.g., `annual_reagent_cost_small_kit_usd`, `annual_carrier_cost_bulk_kit_usd`, `annual_margin_difference_bulk_minus_small_usd`) must appear as a flat key at the assay level. Similarly, `assumptions` must have flat keys like `runs_per_year_small_kit`, NOT nested objects.

**CARRIER COST NOTE**: The instructions don't give an explicit formula for annual carrier cost. Think about it logically: each run likely requires a shipment to each lab. So `annual_carrier_cost = carrier_cost_usd * active_labs * runs_per_year`. However, if the data or verifier suggests otherwise (e.g., carrier cost is a flat per-run cost not multiplied by labs), be prepared to adjust. Start with `carrier_cost_usd * active_labs * runs_per_year`.

Wait — actually re-read the instructions more carefully. The carrier cost formula is NOT explicitly stated. Let me reconsider: it could also be `carrier_cost_usd * runs_per_year` (flat, not per-lab). Check if the numbers make sense after computing. If the verifier fails, try the alternative.

Actually, looking again at the problem structure: reagent costs and revenue are both per-lab, so carrier costs being per-lab too is consistent. Use `carrier_cost_usd * active_labs * runs_per_year`.

## 3. Run the script

```bash
cd /root && python solve.py
```

## 4. Validate outputs

- Verify `/root/reagent_policy_report.json` is valid JSON with the correct schema (flat keys, metadata preserved from template, assays sorted by assay_id)
- Verify `/root/reagent_policy_summary.md` has 4-8 non-empty lines and includes all required elements
- Check that all currency values are rounded to 2 decimal places
- Verify the decision logic: if abs(total_difference) < 7000 → adopt_bulk_kit, else keep_small_kit

## 5. If the verifier test file exists, run it

Check if there's a test file (e.g., `test_outputs.py` or similar) in `/root/` and run it with pytest if found:
```bash
find /root -name 'test_*.py' -o -name '*_test.py' | head -5
# If found, run: python -m pytest <test_file> -v
```

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
Task metadata: author_email=gpt54@example.com, author_name=GPT-5.4, category=financial-analysis, difficulty=medium, tags=[lab-operations, json, csv, template-update, decision-analysis].
Verifier config: timeout_sec=900.0.