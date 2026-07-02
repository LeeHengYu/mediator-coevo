# Task Instruction

Execute the following Python script to produce `/root/refill_analysis.json` and `/root/refill_summary.md`.

```python
import csv
import json
import os

# 1. Read input CSVs
def read_csv(path):
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]
    return rows

wholesale_rows = read_csv('/root/wholesale_price.csv')
vial_rows = read_csv('/root/vial_price.csv')
reimbursement_rows = read_csv('/root/reimbursement.csv')

# 2. Print raw data for debugging
print('=== wholesale_price.csv ===')
for r in wholesale_rows:
    print(r)
print('=== vial_price.csv ===')
for r in vial_rows:
    print(r)
print('=== reimbursement.csv ===')
for r in reimbursement_rows:
    print(r)

# 3. Identify column names (strip whitespace)
def clean_keys(rows):
    return [{k.strip(): v.strip() for k, v in row.items()} for row in rows]

wholesale_rows = clean_keys(wholesale_rows)
vial_rows = clean_keys(vial_rows)
reimbursement_rows = clean_keys(reimbursement_rows)

# 4. Build lookup dicts by medication name
# Detect the medication name column in each CSV
def find_med_col(row):
    for k in row:
        kl = k.lower()
        if 'medication' in kl or 'drug' in kl or 'name' in kl:
            return k
    # fallback: first column
    return list(row.keys())[0]

med_col_w = find_med_col(wholesale_rows[0])
med_col_v = find_med_col(vial_rows[0])
med_col_r = find_med_col(reimbursement_rows[0])

print(f'\nMed columns: wholesale={med_col_w}, vial={med_col_v}, reimb={med_col_r}')

# Find price column in wholesale
def find_col(row, candidates):
    for k in row:
        kl = k.lower()
        for c in candidates:
            if c in kl:
                return k
    return None

price_col = find_col(wholesale_rows[0], ['price_per_1000', 'price per 1000', '1000'])
if not price_col:
    # fallback: any column with 'price' that isn't the med col
    for k in wholesale_rows[0]:
        if 'price' in k.lower() and k != med_col_w:
            price_col = k
            break
print(f'Price col: {price_col}')

# Find vial size and vial price columns
vial_size_col = find_col(vial_rows[0], ['size', 'dram'])
vial_price_col = find_col(vial_rows[0], ['price', 'cost'])
if vial_price_col and vial_size_col and vial_price_col == vial_size_col:
    # need to differentiate
    for k in vial_rows[0]:
        if k != med_col_v and k != vial_size_col and ('price' in k.lower() or 'cost' in k.lower()):
            vial_price_col = k
            break
print(f'Vial size col: {vial_size_col}, Vial price col: {vial_price_col}')

# Find reimbursement column
reimb_col = find_col(reimbursement_rows[0], ['reimbursement', 'reimb', 'amount', 'per_fill', 'per fill'])
if not reimb_col:
    for k in reimbursement_rows[0]:
        if k != med_col_r:
            reimb_col = k
            break
print(f'Reimb col: {reimb_col}')

# Build dicts
def normalize_med(name):
    return name.strip().lower()

wholesale_dict = {}
for r in wholesale_rows:
    med = r[med_col_w]
    wholesale_dict[normalize_med(med)] = {
        'medication': med,
        'price_per_1000': float(r[price_col].replace(',', '').replace('$', ''))
    }

vial_dict = {}
for r in vial_rows:
    med = r[med_col_v]
    size_val = r.get(vial_size_col, '0')
    price_val = r.get(vial_price_col, '0')
    vial_dict[normalize_med(med)] = {
        'vial_size_drams': int(float(size_val.replace(',', '').replace('$', ''))) if size_val else 0,
        'vial_price_usd': float(price_val.replace(',', '').replace('$', '')) if price_val else 0.0
    }

reimb_dict = {}
for r in reimbursement_rows:
    med = r[med_col_r]
    reimb_dict[normalize_med(med)] = {
        'reimbursement': float(r[reimb_col].replace(',', '').replace('$', ''))
    }

print(f'\nWholesale meds: {list(wholesale_dict.keys())}')
print(f'Vial meds: {list(vial_dict.keys())}')
print(f'Reimb meds: {list(reimb_dict.keys())}')

# 5. Constants
PATIENTS = 300
FILLS_90 = 4
FILLS_100 = 3
TABLETS_90 = 90
TABLETS_100 = 100
THRESHOLD = 16000

# 6. Compute per-medication
medications = []

# Use wholesale_dict keys as the canonical medication list
for med_key in wholesale_dict:
    w = wholesale_dict[med_key]
    v = vial_dict.get(med_key, {'vial_size_drams': 0, 'vial_price_usd': 0.0})
    rb = reimb_dict.get(med_key, {'reimbursement': 0.0})

    price_per_1000 = w['price_per_1000']
    vial_price = v['vial_price_usd']
    vial_size = v['vial_size_drams']
    reimb_per_fill = rb['reimbursement']

    # Drug cost: (tablets_per_fill * patients * fills * price_per_1000) / 1000
    annual_drug_cost_90 = round((TABLETS_90 * PATIENTS * FILLS_90 * price_per_1000) / 1000, 2)
    annual_drug_cost_100 = round((TABLETS_100 * PATIENTS * FILLS_100 * price_per_1000) / 1000, 2)

    # Supply cost: vial_price * patients * fills
    annual_supply_cost_90 = round(vial_price * PATIENTS * FILLS_90, 2)
    annual_supply_cost_100 = round(vial_price * PATIENTS * FILLS_100, 2)

    # Reimbursement: reimb_per_fill * fills  (reimb_per_fill is already for 300 patients)
    annual_reimb_90 = round(reimb_per_fill * FILLS_90, 2)
    annual_reimb_100 = round(reimb_per_fill * FILLS_100, 2)

    # Revenue
    annual_rev_90 = round(annual_reimb_90 - annual_drug_cost_90 - annual_supply_cost_90, 2)
    annual_rev_100 = round(annual_reimb_100 - annual_drug_cost_100 - annual_supply_cost_100, 2)

    diff = round(annual_rev_100 - annual_rev_90, 2)

    medications.append({
        'medication': w['medication'],
        'price_per_1000_tablets_usd': price_per_1000,
        'vial_size_drams': vial_size,
        'vial_price_usd': vial_price,
        'reimbursement_per_fill_300_patients_usd': reimb_per_fill,
        'annual_drug_cost_90_day_usd': annual_drug_cost_90,
        'annual_drug_cost_100_day_usd': annual_drug_cost_100,
        'annual_supply_cost_90_day_usd': annual_supply_cost_90,
        'annual_supply_cost_100_day_usd': annual_supply_cost_100,
        'annual_reimbursement_90_day_usd': annual_reimb_90,
        'annual_reimbursement_100_day_usd': annual_reimb_100,
        'annual_revenue_90_day_usd': annual_rev_90,
        'annual_revenue_100_day_usd': annual_rev_100,
        'annual_revenue_difference_100_minus_90_usd': diff
    })

# 7. Totals
total_rev_90 = round(sum(m['annual_revenue_90_day_usd'] for m in medications), 2)
total_rev_100 = round(sum(m['annual_revenue_100_day_usd'] for m in medications), 2)
total_diff = round(total_rev_100 - total_rev_90, 2)
abs_diff = round(abs(total_diff), 2)

print(f'\nTotal rev 90: {total_rev_90}')
print(f'Total rev 100: {total_rev_100}')
print(f'Total diff: {total_diff}')
print(f'Abs diff: {abs_diff}')

# 8. Decision
if abs_diff < THRESHOLD:
    decision = 'switch_to_100_day'
    justification = (f'The absolute total revenue difference is ${abs_diff:,.2f}, '
                     f'which is below the ${THRESHOLD:,} threshold. '
                     f'Switching to 100-day fills is recommended for patient convenience '
                     f'with minimal financial impact.')
else:
    decision = 'keep_90_day'
    justification = (f'The absolute total revenue difference is ${abs_diff:,.2f}, '
                     f'which exceeds the ${THRESHOLD:,} threshold. '
                     f'Keeping 90-day fills is recommended to preserve revenue.')

print(f'Decision: {decision}')

# 9. Build output JSON
output = {
    'assumptions': {
        'patients_per_medication': PATIENTS,
        'fills_per_year_90_day': FILLS_90,
        'fills_per_year_100_day': FILLS_100,
        'tablets_per_fill_90_day': TABLETS_90,
        'tablets_per_fill_100_day': TABLETS_100,
        'switch_threshold_usd': THRESHOLD
    },
    'medications': medications,
    'totals': {
        'total_annual_revenue_90_day_usd': total_rev_90,
        'total_annual_revenue_100_day_usd': total_rev_100,
        'total_annual_revenue_difference_100_minus_90_usd': total_diff,
        'absolute_total_revenue_difference_usd': abs_diff
    },
    'recommendation': {
        'decision': decision,
        'justification': justification
    }
}

with open('/root/refill_analysis.json', 'w') as f:
    json.dump(output, f, indent=2)
print('\nWrote /root/refill_analysis.json')

# 10. Build summary markdown
summary_lines = [
    '# Refill Policy Analysis Summary',
    '',
    f'- Total 90-day annual revenue: ${total_rev_90:,.2f}',
    f'- Total 100-day annual revenue: ${total_rev_100:,.2f}',
    f'- Absolute revenue difference: ${abs_diff:,.2f}',
    f'- Decision: {decision}',
    '',
    justification
]

with open('/root/refill_summary.md', 'w') as f:
    f.write('\n'.join(summary_lines) + '\n')
print('Wrote /root/refill_summary.md')

# 11. Verify outputs exist and print them
print('\n=== refill_analysis.json ===')
with open('/root/refill_analysis.json') as f:
    print(f.read())
print('\n=== refill_summary.md ===')
with open('/root/refill_summary.md') as f:
    print(f.read())
```

Steps:
1. First, inspect the three CSV files to understand their column names: `cat /root/wholesale_price.csv`, `cat /root/vial_price.csv`, `cat /root/reimbursement.csv`. Print the first few lines.
2. Then run the Python script above.
3. After running, verify both output files exist and print their contents.
4. Confirm the JSON has exactly 10 medications, all required keys are present, all currency values are rounded to 2 decimals, and the summary has 4-8 lines including the required fields (total 90-day revenue, total 100-day revenue, absolute difference, and decision slug).
5. If any CSV column names don't match the auto-detection logic, adjust the column name references and re-run.

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

Task-local resources are available under `environment/skills`: business-model-math-validation, loyalty-modeling, pharmacy-supply-chain, recursive-generosity-protocol, value-analysis.

# Verifier Contract

Success is judged by the SkillFlow verifier for this task.
Do not bypass, remove, or weaken verifier scripts, tests, fixtures, or expected-output checks.
Run the provided tests or verifier command when practical before finalizing.
Task metadata: author_email=codex@example.com, author_name=Codex, category=financial-analysis, difficulty=medium, tags=[pharmacy, unit-economics, cost-analysis, json, verification].
Verifier config: timeout_sec=900.0.