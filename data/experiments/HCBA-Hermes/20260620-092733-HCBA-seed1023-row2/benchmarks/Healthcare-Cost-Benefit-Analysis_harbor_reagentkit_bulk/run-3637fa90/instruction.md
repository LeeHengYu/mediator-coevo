# Task Instruction

## Task: Healthcare Cost-Benefit Analysis – Harbor ReagentKit Bulk

You must produce two output files by reading and processing five input files according to precise rules. Follow every step carefully.

### Step 1: Read all input files

```bash
cat /root/assay_manifest.json
cat /root/carrier_cost.csv
cat /root/billing.csv
cat /root/lab_overrides.csv
cat /root/report_template.json
```

Read and understand the structure of each file before proceeding.

### Step 2: Write a Python script `/root/solve.py` that does the following

```python
import json, csv, math
from collections import defaultdict

# 1. Load all input files
with open('/root/assay_manifest.json') as f:
    manifest = json.load(f)
with open('/root/carrier_cost.csv') as f:
    carrier_rows = list(csv.DictReader(f))
with open('/root/billing.csv') as f:
    billing_rows = list(csv.DictReader(f))
with open('/root/lab_overrides.csv') as f:
    override_rows = list(csv.DictReader(f))
with open('/root/report_template.json') as f:
    template = json.load(f)

# 2. Build carrier cost lookup: carrier_type -> carrier_cost_usd
carrier_lookup = {}
for row in carrier_rows:
    carrier_lookup[row['carrier_type'].strip()] = float(row['carrier_cost_usd'])

# 3. Filter in-scope assays from manifest
# manifest could be a list or dict with assays key - inspect structure
if isinstance(manifest, dict) and 'assays' in manifest:
    assays_list = manifest['assays']
elif isinstance(manifest, list):
    assays_list = manifest
else:
    # try to find the assays
    assays_list = manifest.get('assays', manifest)

in_scope = [a for a in assays_list if a.get('in_scope') == True or a.get('in_scope') == 'true']

# 4. Build alias map: for each in-scope assay, map assay_name and all aliases to assay_id
alias_to_assay = {}
for a in in_scope:
    aid = a['assay_id']
    alias_to_assay[a['assay_name'].strip().lower()] = aid
    for alias in a.get('aliases', []):
        alias_to_assay[alias.strip().lower()] = aid

# 5. Resolve billing rows
# Filter active billing rows, match to assay_id via alias map
# Keep latest effective_month per assay_id
billing_by_assay = {}
for row in billing_rows:
    if row.get('is_active', '').strip().lower() != 'true':
        continue
    label = row['assay_label'].strip().lower()
    aid = alias_to_assay.get(label)
    if aid is None:
        continue
    em = row['effective_month'].strip()
    if aid not in billing_by_assay or em > billing_by_assay[aid]['effective_month']:
        billing_by_assay[aid] = {'effective_month': em, 'payment_per_run_per_lab_usd': float(row['payment_per_run_per_lab_usd'])}

# 6. Resolve lab overrides
# Filter approved rows, keep highest revision per assay_id
override_by_assay = {}
for row in override_rows:
    if row.get('status', '').strip().lower() != 'approved':
        continue
    aid = row['assay_id'].strip()
    rev = int(row['revision'])
    if aid not in override_by_assay or rev > override_by_assay[aid]['revision']:
        override_by_assay[aid] = {'revision': rev, 'active_labs': int(row['active_labs'])}

# 7. Process each in-scope assay
results = []
for a in in_scope:
    aid = a['assay_id']
    aname = a['assay_name']
    
    # Active labs
    if aid in override_by_assay:
        active_labs = override_by_assay[aid]['active_labs']
    else:
        active_labs = int(a['default_active_labs'])
    
    reagent_price = float(a['reagent_price_per_1000_tests_usd'])
    carrier_type = a['carrier_type'].strip()
    carrier_cost = carrier_lookup[carrier_type]
    
    # Billing
    payment = billing_by_assay[aid]['payment_per_run_per_lab_usd']
    
    tests_small = int(a['tests_per_lab_per_run_small'])
    tests_bulk = int(a['tests_per_lab_per_run_bulk'])
    
    runs_small = 24
    runs_bulk = 12
    
    # Annual revenue
    rev_small = payment * active_labs * runs_small
    rev_bulk = payment * active_labs * runs_bulk
    
    # Annual reagent cost
    reagent_small = reagent_price * active_labs * tests_small * runs_small / 1000
    reagent_bulk = reagent_price * active_labs * tests_bulk * runs_bulk / 1000
    
    # Annual carrier cost - NOTE: The task mentions carrier_cost_usd and annual_carrier_cost fields.
    # The carrier_cost.csv has carrier_cost_usd per shipment/run.
    # annual_carrier_cost = carrier_cost_usd * active_labs * runs_per_year
    carrier_small = carrier_cost * active_labs * runs_small
    carrier_bulk = carrier_cost * active_labs * runs_bulk
    
    # Annual margin
    margin_small = rev_small - reagent_small - carrier_small
    margin_bulk = rev_bulk - reagent_bulk - carrier_bulk
    
    diff = margin_bulk - margin_small
    
    results.append({
        'assay_id': aid,
        'assay_name': aname,
        'active_labs': active_labs,
        'reagent_price_per_1000_tests_usd': round(reagent_price, 2),
        'carrier_type': carrier_type,
        'carrier_cost_usd': round(carrier_cost, 2),
        'payment_per_run_per_lab_usd': round(payment, 2),
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
results.sort(key=lambda x: x['assay_id'])

# Totals
total_margin_small = round(sum(r['annual_margin_small_kit_usd'] for r in results), 2)
total_margin_bulk = round(sum(r['annual_margin_bulk_kit_usd'] for r in results), 2)
total_diff = round(sum(r['annual_margin_difference_bulk_minus_small_usd'] for r in results), 2)
abs_diff = round(abs(total_diff), 2)

# Decision
if abs_diff < 7000:
    decision = 'adopt_bulk_kit'
    justification = f'Absolute total margin difference ${abs_diff} is below the $7000 threshold, so bulk-kit adoption is recommended.'
else:
    decision = 'keep_small_kit'
    justification = f'Absolute total margin difference ${abs_diff} meets or exceeds the $7000 threshold, so the small-kit cadence should be retained.'

# Build output JSON
output = {
    'metadata': template['metadata'],
    'analysis': {
        'assumptions': {
            'runs_per_year_small_kit': 24,
            'runs_per_year_bulk_kit': 12,
            'switch_threshold_usd': 7000,
            'lab_override_rule': 'highest approved revision per assay_id, else default_active_labs',
            'billing_rule': 'latest active effective_month per assay'
        },
        'assays': results,
        'totals': {
            'total_annual_margin_small_kit_usd': total_margin_small,
            'total_annual_margin_bulk_kit_usd': total_margin_bulk,
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
    json.dump(output, f, indent=2)

# Write summary markdown
lines = [
    '# Reagent Policy Summary',
    f'Total annual margin (small-kit): ${total_margin_small:,.2f}',
    f'Total annual margin (bulk-kit): ${total_margin_bulk:,.2f}',
    f'Absolute margin difference: ${abs_diff:,.2f}',
    f'Recommendation: {decision}',
    f'The analysis evaluated {len(results)} in-scope assays comparing 24 small-kit runs/year versus 12 bulk-kit runs/year.'
]
with open('/root/reagent_policy_summary.md', 'w') as f:
    f.write('\n'.join(lines) + '\n')

print('Done. Files written.')
print(f'Total small-kit margin: {total_margin_small}')
print(f'Total bulk-kit margin: {total_margin_bulk}')
print(f'Total difference: {total_diff}')
print(f'Absolute difference: {abs_diff}')
print(f'Decision: {decision}')
```

**IMPORTANT**: Before writing the script, first inspect all five input files to understand their exact structure (column names, field names, data types, whether `in_scope` is boolean or string, etc.). Adapt the script accordingly. The script above is a template — you MUST adjust it based on the actual file contents.

Specific things to check:
- Is `assay_manifest.json` a list or an object with an `assays` key?
- What are the exact column headers in each CSV (watch for whitespace)?
- Is `in_scope` a JSON boolean (`true`) or a string (`"true"`)?
- Is `is_active` in billing.csv a string or boolean?
- Does `carrier_cost.csv` have a header row?
- What does `lab_overrides.csv` look like — column names for `assay_id`, `status`, `revision`, `active_labs`?
- What does `report_template.json` contain for `metadata`?

### Step 3: Run the script

```bash
python3 /root/solve.py
```

### Step 4: Validate outputs

1. Verify JSON is valid:
```bash
python3 -c "import json; d=json.load(open('/root/reagent_policy_report.json')); print('Keys:', list(d.keys())); print('Assay count:', len(d['analysis']['assays'])); print('Totals:', d['analysis']['totals']); print('Decision:', d['analysis']['recommendation']['decision']); print('Metadata:', d['metadata'])"
```

2. Verify the summary markdown:
```bash
cat /root/reagent_policy_summary.md
```

3. Check that:
   - `metadata` matches the template exactly
   - `assays` are sorted by `assay_id` ascending
   - All currency values have exactly 2 decimal places
   - The decision slug is exactly `adopt_bulk_kit` or `keep_small_kit`
   - The summary has 4-8 non-empty lines and includes all required values
   - The summary contains the exact decision slug

### Step 5: Cross-check one assay manually

Pick the first assay in the sorted output and manually verify:
- active_labs resolution (override vs default)
- billing row selection (latest active effective_month)
- revenue, reagent cost, carrier cost, and margin calculations for both models
- margin difference

If anything is wrong, fix the script and re-run.

### Key Pitfalls to Avoid
- Do NOT include out-of-scope assays (in_scope must be true)
- Do NOT include inactive billing rows
- Do NOT forget to match billing by aliases as well as assay_name
- The carrier cost formula is: `carrier_cost_usd * active_labs * runs_per_year` (this is the annual carrier cost)
- Make sure rounding to 2 decimals happens on the final values, not intermediate ones
- The totals should be computed from the rounded per-assay values to match the schema
- `abs(total_difference) < 7000` means strictly less than 7000 → adopt_bulk_kit; otherwise keep_small_kit

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