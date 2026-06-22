# Task Instruction

Execute the following steps in order:

1. **Inspect the input files** to understand their structure:
```bash
cat /root/ingredient_cost.csv
cat /root/card_cost.csv
cat /root/reimbursement.csv
```

2. **Inspect the test file** to understand exact verifier expectations:
```bash
cat /root/tests/test_outputs.py
```

3. **Create and run a Python script** `/root/solve.py` that does the following:

```python
import csv
import json
import os

# Read ingredient_cost.csv
ingredient_costs = {}
with open('/root/ingredient_cost.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        med = row['medication'].strip()
        ingredient_costs[med] = float(row['price_per_1000_capsules_usd'])

# Read card_cost.csv
card_costs = {}
with open('/root/card_cost.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        count = int(row['blister_card_count'])
        card_costs[count] = float(row['card_cost_usd'])

# Read reimbursement.csv
reimbursements = {}
with open('/root/reimbursement.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        med = row['medication'].strip()
        reimbursements[med] = float(row['reimbursement_per_cycle_180_patients_usd'])

# Constants
patients = 180
capsules_per_day = 2
fills_28 = 12
fills_56 = 6
caps_per_fill_28 = 56
caps_per_fill_56 = 112
threshold = 9000

medications = []
for med in sorted(ingredient_costs.keys()):
    price_per_1000 = ingredient_costs[med]
    reimb_per_cycle = reimbursements[med]
    
    # Determine blister card count for this medication
    # 28-day: 56 capsules per fill; 56-day: 112 capsules per fill
    # We need to find the card_cost for the appropriate blister_card_count
    # The blister_card_count in card_cost.csv should match capsules_per_fill
    # Actually, let's check what blister_card_counts are available
    # For 28-day model, blister_card_count = 56; for 56-day, blister_card_count = 112
    # But the card_cost might be per-medication. Let me re-check.
    # Actually the task says: "matched by blister_card_count"
    # So card_cost.csv has rows with blister_card_count and card_cost_usd
    # 28-day uses blister_card_count=56, 56-day uses blister_card_count=112
    
    # Drug cost = (capsules_per_fill * fills * patients * price_per_1000) / 1000
    annual_drug_cost_28 = round((caps_per_fill_28 * fills_28 * patients * price_per_1000) / 1000, 2)
    annual_drug_cost_56 = round((caps_per_fill_56 * fills_56 * patients * price_per_1000) / 1000, 2)
    
    # Packaging cost = card_cost * patients * fills
    # Need to figure out blister_card_count mapping
    # For 28-day: blister_card_count = caps_per_fill_28 = 56
    # For 56-day: blister_card_count = caps_per_fill_56 = 112
    card_cost_28 = card_costs.get(caps_per_fill_28, 0.0)
    card_cost_56 = card_costs.get(caps_per_fill_56, 0.0)
    
    annual_packaging_28 = round(card_cost_28 * patients * fills_28, 2)
    annual_packaging_56 = round(card_cost_56 * patients * fills_56, 2)
    
    # Reimbursement
    annual_reimb_28 = round(reimb_per_cycle * fills_28, 2)
    annual_reimb_56 = round(reimb_per_cycle * fills_56, 2)
    
    # Margins
    annual_margin_28 = round(annual_reimb_28 - annual_drug_cost_28 - annual_packaging_28, 2)
    annual_margin_56 = round(annual_reimb_56 - annual_drug_cost_56 - annual_packaging_56, 2)
    margin_diff = round(annual_margin_56 - annual_margin_28, 2)
    
    # Determine which blister_card_count to report
    # The schema has a single blister_card_count and card_cost_usd per medication
    # This likely refers to the base card count. Let me check card_cost.csv structure.
    # If card_cost.csv has medication-specific rows, we need per-med. Otherwise it's global.
    # The schema shows one blister_card_count per med entry - might be the 28-day one.
    # Let's use 56 (28-day caps_per_fill) as the base blister_card_count
    
    medications.append({
        'medication': med,
        'price_per_1000_capsules_usd': price_per_1000,
        'blister_card_count': caps_per_fill_28,
        'card_cost_usd': card_cost_28,
        'reimbursement_per_cycle_180_patients_usd': reimb_per_cycle,
        'annual_drug_cost_28_day_usd': annual_drug_cost_28,
        'annual_drug_cost_56_day_usd': annual_drug_cost_56,
        'annual_packaging_cost_28_day_usd': annual_packaging_28,
        'annual_packaging_cost_56_day_usd': annual_packaging_56,
        'annual_reimbursement_28_day_usd': annual_reimb_28,
        'annual_reimbursement_56_day_usd': annual_reimb_56,
        'annual_margin_28_day_usd': annual_margin_28,
        'annual_margin_56_day_usd': annual_margin_56,
        'annual_margin_difference_56_minus_28_usd': margin_diff
    })

total_margin_28 = round(sum(m['annual_margin_28_day_usd'] for m in medications), 2)
total_margin_56 = round(sum(m['annual_margin_56_day_usd'] for m in medications), 2)
total_diff = round(total_margin_56 - total_margin_28, 2)
abs_diff = round(abs(total_diff), 2)

if abs_diff < threshold:
    decision = 'convert_to_56_day'
    justification = f'The absolute total margin difference of ${abs_diff:,.2f} is below the ${threshold:,.2f} threshold, so converting to 56-day cycles is recommended.'
else:
    decision = 'keep_28_day'
    justification = f'The absolute total margin difference of ${abs_diff:,.2f} exceeds the ${threshold:,.2f} threshold, so keeping 28-day cycles is recommended.'

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

# Write summary with comma-formatted currency values
lines = [
    '# Syncpack Analysis Summary',
    f'Total 28-day annual margin: ${total_margin_28:,.2f}',
    f'Total 56-day annual margin: ${total_margin_56:,.2f}',
    f'Total margin difference (56-day minus 28-day): ${total_diff:,.2f}',
    f'Absolute total margin difference: ${abs_diff:,.2f}',
    f'Decision: {decision}',
    justification
]

with open('/root/syncpack_summary.md', 'w') as f:
    f.write('\n'.join(lines) + '\n')

print('Done. Files written.')
print(f'Total 28-day margin: {total_margin_28}')
print(f'Total 56-day margin: {total_margin_56}')
print(f'Total difference: {total_diff}')
print(f'Absolute difference: {abs_diff}')
print(f'Decision: {decision}')
```

**IMPORTANT:** Before running this script, first inspect the CSV files and the test file. The script above makes assumptions about card_cost.csv structure. After inspecting the files:

- If `card_cost.csv` has medication-specific rows (not just blister_card_count rows), adjust the lookup logic accordingly.
- If `card_cost.csv` maps blister_card_count to card_cost_usd globally, the script logic is correct but you need to verify which blister_card_counts are available (they should include 56 and 112, or whatever the capsules_per_fill values are).
- If the blister_card_count values in card_cost.csv don't match 56/112, you may need to interpret the mapping differently (e.g., cards per fill might be capsules_per_fill / blister_card_count).
- Check the `reimbursement.csv` to see if it has separate rows for 28-day and 56-day cycles or a single reimbursement_per_cycle value per medication.

After inspecting the CSVs and test file, adjust the script if needed, then run it.

4. **Verify the output:**
```bash
cat /root/syncpack_analysis.json
cat /root/syncpack_summary.md
```

5. **Run the test suite:**
```bash
cd /root && python -m pytest tests/ -v
```

6. If any tests fail, read the error messages carefully, fix the issues, and re-run. Pay special attention to:
   - The `assumptions` block must have exactly these keys: `patients_per_medication`, `fills_per_year_28_day`, `fills_per_year_56_day`, `capsules_per_fill_28_day`, `capsules_per_fill_56_day`, `switch_threshold_usd`
   - Currency values in the markdown summary MUST use comma separators (e.g., `42,908.83` not `42908.83`). Use `f'{value:,.2f}'` formatting.
   - All currency values in JSON must be rounded to 2 decimal places.
   - Medications array must be sorted alphabetically by medication name.
   - The `blister_card_count` and `card_cost_usd` fields per medication in the JSON need to reflect the actual data from card_cost.csv - inspect carefully how medications map to card counts.

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