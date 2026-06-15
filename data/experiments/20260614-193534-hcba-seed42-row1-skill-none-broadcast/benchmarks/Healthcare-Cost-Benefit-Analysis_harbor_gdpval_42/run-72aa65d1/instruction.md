# Task Instruction

Execute the following steps in order:

1. **Inspect input files.** Read and display the contents of:
   - `/root/wholesale_price.csv`
   - `/root/vial_price.csv`
   - `/root/reimbursement.csv`
   Note the exact column names and the number of medications (should be 10).

2. **Write and run a Python script** (`/root/solve.py`) that does the following:

```python
import csv, json, math

# Read wholesale_price.csv
with open('/root/wholesale_price.csv') as f:
    wholesale = list(csv.DictReader(f))

# Read vial_price.csv
with open('/root/vial_price.csv') as f:
    vials = list(csv.DictReader(f))

# Read reimbursement.csv
with open('/root/reimbursement.csv') as f:
    reimb = list(csv.DictReader(f))

# Build lookup dicts keyed by medication name (strip whitespace)
# Inspect column names first and adapt accordingly
print('wholesale columns:', wholesale[0].keys() if wholesale else 'EMPTY')
print('vial columns:', vials[0].keys() if vials else 'EMPTY')
print('reimb columns:', reimb[0].keys() if reimb else 'EMPTY')
print('wholesale rows:', len(wholesale))
print('vial rows:', len(vials))
print('reimb rows:', len(reimb))

# Identify the medication name column in each file
def find_med_col(row):
    for k in row.keys():
        if 'med' in k.lower() or 'drug' in k.lower() or 'name' in k.lower():
            return k
    return list(row.keys())[0]

med_col_w = find_med_col(wholesale[0])
med_col_v = find_med_col(vials[0])
med_col_r = find_med_col(reimb[0])

# Find price_per_1000 column
def find_col(row, substr):
    for k in row.keys():
        if substr in k.lower():
            return k
    return None

price_col = find_col(wholesale[0], 'price_per_1000') or find_col(wholesale[0], 'price')
vial_size_col = find_col(vials[0], 'size') or find_col(vials[0], 'dram')
vial_price_col = find_col(vials[0], 'price')
reimb_col = find_col(reimb[0], 'reimbursement') or find_col(reimb[0], 'per_fill') or find_col(reimb[0], 'usd')

print(f'Using columns: med_w={med_col_w}, price={price_col}, med_v={med_col_v}, vial_size={vial_size_col}, vial_price={vial_price_col}, med_r={med_col_r}, reimb={reimb_col}')

# Build dicts
wholesale_dict = {row[med_col_w].strip(): float(row[price_col]) for row in wholesale}
vial_dict = {row[med_col_v].strip(): {'size': row.get(vial_size_col, ''), 'price': float(row[vial_price_col])} for row in vials}
reimb_dict = {row[med_col_r].strip(): float(row[reimb_col]) for row in reimb}

patients = 300
fills_90 = 4
fills_100 = 3
tablets_90 = 90
tablets_100 = 100
threshold = 16000

medications_out = []
med_names = list(wholesale_dict.keys())  # use wholesale ordering

for med in med_names:
    p1000 = wholesale_dict[med]
    vial_info = vial_dict[med]
    vial_price = vial_info['price']
    vial_size_str = vial_info['size']
    # Try to parse vial size as int
    try:
        vial_size = int(vial_size_str)
    except:
        vial_size = 0
    reimb_per_fill = reimb_dict[med]

    # Drug cost per fill = (tablets_per_fill / 1000) * price_per_1000 * patients
    # Annual drug cost = drug_cost_per_fill * fills_per_year
    annual_drug_90 = round((tablets_90 / 1000) * p1000 * patients * fills_90, 2)
    annual_drug_100 = round((tablets_100 / 1000) * p1000 * patients * fills_100, 2)

    # Supply cost: vial_price per patient per fill
    annual_supply_90 = round(vial_price * patients * fills_90, 2)
    annual_supply_100 = round(vial_price * patients * fills_100, 2)

    # Reimbursement: reimb_per_fill is for 300 patients per fill
    annual_reimb_90 = round(reimb_per_fill * fills_90, 2)
    annual_reimb_100 = round(reimb_per_fill * fills_100, 2)

    # Revenue
    annual_rev_90 = round(annual_reimb_90 - annual_drug_90 - annual_supply_90, 2)
    annual_rev_100 = round(annual_reimb_100 - annual_drug_100 - annual_supply_100, 2)
    diff = round(annual_rev_100 - annual_rev_90, 2)

    medications_out.append({
        'medication': med,
        'price_per_1000_tablets_usd': p1000,
        'vial_size_drams': vial_size,
        'vial_price_usd': vial_price,
        'reimbursement_per_fill_300_patients_usd': reimb_per_fill,
        'annual_drug_cost_90_day_usd': annual_drug_90,
        'annual_drug_cost_100_day_usd': annual_drug_100,
        'annual_supply_cost_90_day_usd': annual_supply_90,
        'annual_supply_cost_100_day_usd': annual_supply_100,
        'annual_reimbursement_90_day_usd': annual_reimb_90,
        'annual_reimbursement_100_day_usd': annual_reimb_100,
        'annual_revenue_90_day_usd': annual_rev_90,
        'annual_revenue_100_day_usd': annual_rev_100,
        'annual_revenue_difference_100_minus_90_usd': diff
    })

total_rev_90 = round(sum(m['annual_revenue_90_day_usd'] for m in medications_out), 2)
total_rev_100 = round(sum(m['annual_revenue_100_day_usd'] for m in medications_out), 2)
total_diff = round(total_rev_100 - total_rev_90, 2)
abs_diff = round(abs(total_diff), 2)

if abs_diff < threshold:
    decision = 'switch_to_100_day'
    justification = f'The absolute total revenue difference of ${abs_diff:.2f} is below the ${threshold:.2f} threshold, so switching to 100-day fills is recommended.'
else:
    decision = 'keep_90_day'
    justification = f'The absolute total revenue difference of ${abs_diff:.2f} exceeds the ${threshold:.2f} threshold, so keeping 90-day fills is recommended.'

output = {
    'assumptions': {
        'patients_per_medication': patients,
        'fills_per_year_90_day': fills_90,
        'fills_per_year_100_day': fills_100,
        'tablets_per_fill_90_day': tablets_90,
        'tablets_per_fill_100_day': tablets_100,
        'switch_threshold_usd': threshold
    },
    'medications': medications_out,
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

print('JSON written.')
print(f'Total 90-day revenue: ${total_rev_90:.2f}')
print(f'Total 100-day revenue: ${total_rev_100:.2f}')
print(f'Absolute difference: ${abs_diff:.2f}')
print(f'Decision: {decision}')

# Write summary markdown
with open('/root/refill_summary.md', 'w') as f:
    f.write('# Refill Policy Analysis Summary\n\n')
    f.write(f'- Total 90-day annual revenue: ${total_rev_90:,.2f}\n')
    f.write(f'- Total 100-day annual revenue: ${total_rev_100:,.2f}\n')
    f.write(f'- Absolute revenue difference: ${abs_diff:,.2f}\n')
    f.write(f'- Recommendation: **{decision}**\n')

print('Summary written.')
```

3. **Run the script:** `python /root/solve.py`

4. **Validate outputs:**
   - Read `/root/refill_analysis.json` and confirm it has the correct schema: `assumptions`, `medications` (list of 10), `totals`, `recommendation`.
   - Confirm all `_usd` fields are rounded to 2 decimal places.
   - Read `/root/refill_summary.md` and confirm it has 4-8 lines, includes total 90-day revenue, total 100-day revenue, absolute difference, and the exact decision slug.
   - If the column names in the CSVs don't match the expected patterns, adapt the column detection logic and re-run.

5. **If a test script exists** (e.g., `/root/test_output.py`), run it: `pytest /root/test_output.py -v` and confirm all tests pass. If tests fail, inspect the error messages, fix the issue, and re-run.

**Key formulas to double-check:**
- `annual_drug_cost = (tablets_per_fill / 1000) * price_per_1000_tablets * 300 * fills_per_year`
- `annual_supply_cost = vial_price * 300 * fills_per_year`
- `annual_reimbursement = reimbursement_per_fill_300_patients * fills_per_year`
- `annual_revenue = annual_reimbursement - annual_drug_cost - annual_supply_cost`
- Decision: if `abs(total_difference) < 16000` → `switch_to_100_day`, else `keep_90_day`

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