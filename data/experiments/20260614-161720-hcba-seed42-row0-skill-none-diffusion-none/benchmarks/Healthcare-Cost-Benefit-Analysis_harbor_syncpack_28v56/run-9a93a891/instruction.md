# Task Instruction

Execute the following steps exactly:

1. **Inspect the input files** to understand their structure:
```bash
cat /root/ingredient_cost.csv
cat /root/card_cost.csv
cat /root/reimbursement.csv
```

2. **Create and run a Python script** `/root/solve.py` that does the following:

```python
import csv, json, math

# Read ingredient_cost.csv
with open('/root/ingredient_cost.csv') as f:
    reader = csv.DictReader(f)
    ingredients = list(reader)

# Read card_cost.csv
with open('/root/card_cost.csv') as f:
    reader = csv.DictReader(f)
    cards = list(reader)

# Read reimbursement.csv
with open('/root/reimbursement.csv') as f:
    reader = csv.DictReader(f)
    reimbursements = list(reader)

# Build lookup: blister_card_count -> card_cost_usd
card_lookup = {}
for row in cards:
    card_lookup[int(row['blister_card_count'])] = float(row['card_cost_usd'])

# Build lookup: medication -> reimbursement_per_cycle_180_patients_usd
reimb_lookup = {}
for row in reimbursements:
    reimb_lookup[row['medication'].strip()] = float(row['reimbursement_per_cycle_180_patients_usd'])

# Constants
patients = 180
fills_28 = 12
fills_56 = 6
caps_per_fill_28 = 56
caps_per_fill_56 = 112
threshold = 9000

medications = []
for row in ingredients:
    med_name = row['medication'].strip()
    price_per_1000 = float(row['price_per_1000_capsules_usd'])
    blister_card_count = int(row['blister_card_count'])
    
    # Card cost for 28-day uses the blister_card_count from ingredient_cost.csv
    card_cost_28 = card_lookup[blister_card_count]
    # For 56-day, the blister_card_count doubles (56-day needs 2x the cards)
    card_cost_56 = card_lookup.get(blister_card_count * 2, card_lookup.get(blister_card_count, 0))
    # NOTE: If there's no 2x entry, we may need to just use 2 cards per fill.
    # Let's check what card_cost.csv actually has and decide.
    
    # Reimbursement
    reimb_per_cycle = reimb_lookup[med_name]
    
    # Annual drug cost: (capsules_per_fill * fills * patients * price_per_1000) / 1000
    annual_drug_cost_28 = round((caps_per_fill_28 * fills_28 * patients * price_per_1000) / 1000, 2)
    annual_drug_cost_56 = round((caps_per_fill_56 * fills_56 * patients * price_per_1000) / 1000, 2)
    
    # Annual packaging cost: card_cost * patients * fills
    annual_packaging_cost_28 = round(card_cost_28 * patients * fills_28, 2)
    annual_packaging_cost_56 = round(card_cost_56 * patients * fills_56, 2)
    
    # Annual reimbursement: reimbursement_per_cycle * fills
    annual_reimb_28 = round(reimb_per_cycle * fills_28, 2)
    annual_reimb_56 = round(reimb_per_cycle * fills_56, 2)
    
    # Annual margin
    annual_margin_28 = round(annual_reimb_28 - annual_drug_cost_28 - annual_packaging_cost_28, 2)
    annual_margin_56 = round(annual_reimb_56 - annual_drug_cost_56 - annual_packaging_cost_56, 2)
    
    diff = round(annual_margin_56 - annual_margin_28, 2)
    
    medications.append({
        'medication': med_name,
        'price_per_1000_capsules_usd': price_per_1000,
        'blister_card_count': blister_card_count,
        'card_cost_usd': card_cost_28,
        'reimbursement_per_cycle_180_patients_usd': reimb_per_cycle,
        'annual_drug_cost_28_day_usd': annual_drug_cost_28,
        'annual_drug_cost_56_day_usd': annual_drug_cost_56,
        'annual_packaging_cost_28_day_usd': annual_packaging_cost_28,
        'annual_packaging_cost_56_day_usd': annual_packaging_cost_56,
        'annual_reimbursement_28_day_usd': annual_reimb_28,
        'annual_reimbursement_56_day_usd': annual_reimb_56,
        'annual_margin_28_day_usd': annual_margin_28,
        'annual_margin_56_day_usd': annual_margin_56,
        'annual_margin_difference_56_minus_28_usd': diff
    })

# Sort alphabetically by medication
medications.sort(key=lambda x: x['medication'])

# Totals
total_margin_28 = round(sum(m['annual_margin_28_day_usd'] for m in medications), 2)
total_margin_56 = round(sum(m['annual_margin_56_day_usd'] for m in medications), 2)
total_diff = round(sum(m['annual_margin_difference_56_minus_28_usd'] for m in medications), 2)
abs_diff = round(abs(total_diff), 2)

# Decision
if abs_diff < 9000:
    decision = 'convert_to_56_day'
    justification = f'The absolute total margin difference of ${abs_diff} is below the ${threshold} threshold, so converting to 56-day cycles is recommended.'
else:
    decision = 'keep_28_day'
    justification = f'The absolute total margin difference of ${abs_diff} exceeds the ${threshold} threshold, so keeping 28-day cycles is recommended.'

result = {
    'assumptions': {
        'patients_per_medication': patients,
        'fills_per_year_28_day': fills_28,
        'fills_per_year_56_day': fills_56,
        'capsules_per_fill_28_day': caps_per_fill_28,
        'capsules_per_fill_56_day': caps_per_fill_56,
        'switch_threshold_usd': threshold
    },
    'medications': medications,
    'totals': {
        'total_annual_margin_28_day_usd': total_margin_28,
        'total_annual_margin_56_day_usd': total_margin_56,
        'total_annual_margin_difference_56_minus_28_usd': total_diff,
        'absolute_total_margin_difference_usd': abs_diff
    },
    'recommendation': {
        'decision': decision,
        'justification': justification
    }
}

with open('/root/syncpack_analysis.json', 'w') as f:
    json.dump(result, f, indent=2)

# Write summary markdown
with open('/root/syncpack_summary.md', 'w') as f:
    f.write('# Syncpack Analysis Summary\n')
    f.write(f'\n')
    f.write(f'Total 28-day annual margin: ${total_margin_28}\n')
    f.write(f'Total 56-day annual margin: ${total_margin_56}\n')
    f.write(f'Absolute difference: ${abs_diff}\n')
    f.write(f'\n')
    f.write(f'Decision: {decision}\n')

print('Done. Outputs written.')
print(json.dumps(result, indent=2))
```

**IMPORTANT — Before running the script**, first inspect the CSV files. Then consider these packaging cost details:

- The `card_cost.csv` file maps `blister_card_count` to `card_cost_usd`.
- For the **28-day model**, each fill uses the `blister_card_count` listed in `ingredient_cost.csv` for that medication, and the cost is looked up from `card_cost.csv`.
- For the **56-day model**, the fill is double the capsules. The task says packaging cost uses `card_cost_usd` from `card_cost.csv` per patient per fill, matched by `blister_card_count`. Since 56-day fills have 112 capsules (double), check if `card_cost.csv` has an entry for double the blister card count. If it does, use that cost. If `card_cost.csv` only has one row per medication or the same blister card count entries, then the 56-day fill likely uses the **same card cost per fill** (the task says "per patient per fill, matched by blister_card_count" — if the blister_card_count doesn't change between models, the card_cost_usd is the same, but fewer fills means lower annual packaging cost).

**After inspecting the CSVs**, adjust the script logic if needed before running. The key question is whether `card_cost.csv` has entries for both the 28-day and 56-day blister card counts, or just one set.

3. **Run the script**:
```bash
python3 /root/solve.py
```

4. **Validate the outputs**:
```bash
python3 -c "
import json
with open('/root/syncpack_analysis.json') as f:
    d = json.load(f)
assert set(d.keys()) == {'assumptions', 'medications', 'totals', 'recommendation'}, f'Root keys: {set(d.keys())}'
assert 'patients_per_medication' in d['assumptions']
assert len(d['medications']) > 0
for m in d['medications']:
    for k in ['medication','price_per_1000_capsules_usd','blister_card_count','card_cost_usd',
              'reimbursement_per_cycle_180_patients_usd','annual_drug_cost_28_day_usd',
              'annual_drug_cost_56_day_usd','annual_packaging_cost_28_day_usd',
              'annual_packaging_cost_56_day_usd','annual_reimbursement_28_day_usd',
              'annual_reimbursement_56_day_usd','annual_margin_28_day_usd',
              'annual_margin_56_day_usd','annual_margin_difference_56_minus_28_usd']:
        assert k in m, f'Missing key {k} in medication {m.get("medication","?")}'  
assert 'total_annual_margin_28_day_usd' in d['totals']
assert 'total_annual_margin_56_day_usd' in d['totals']
assert 'total_annual_margin_difference_56_minus_28_usd' in d['totals']
assert 'absolute_total_margin_difference_usd' in d['totals']
assert d['recommendation']['decision'] in ('convert_to_56_day', 'keep_28_day')
print('JSON schema validation passed.')

# Check medications sorted alphabetically
names = [m['medication'] for m in d['medications']]
assert names == sorted(names), f'Not sorted: {names}'
print('Sort order OK.')

# Check summary
with open('/root/syncpack_summary.md') as f:
    text = f.read()
lines = [l for l in text.strip().split('\n') if l.strip()]
assert 4 <= len(lines) <= 8, f'Summary has {len(lines)} non-empty lines'
assert d['recommendation']['decision'] in text
print('Summary validation passed.')
print('All checks passed.')
"
```

5. If any check fails, fix the issue and re-run. Pay special attention to:
   - The packaging cost logic for 56-day (inspect `card_cost.csv` carefully)
   - All JSON keys matching exactly as specified in the schema
   - Alphabetical sort of medications array
   - Rounding to 2 decimal places
   - The decision threshold logic: `abs(total_difference) < 9000` → `convert_to_56_day`, otherwise `keep_28_day`

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
Task metadata: author_email=gpt54@example.com, author_name=GPT-5.4, category=financial-analysis, difficulty=medium, tags=[med-sync, packaging, csv, json, decision-analysis].
Verifier config: timeout_sec=900.0.