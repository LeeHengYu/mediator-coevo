# Task Instruction

Execute the following steps in order:

1. **Inspect the input files** to understand their structure:
```bash
cat /root/ingredient_cost.csv
cat /root/card_cost.csv
cat /root/reimbursement.csv
```

2. **Create `/root/solve.py`** with the following logic:

```python
import csv
import json

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
fills_28 = 12
fills_56 = 6
caps_28 = 56
caps_56 = 112
threshold = 9000

medications = sorted(ingredient_costs.keys())

med_results = []
total_margin_28 = 0.0
total_margin_56 = 0.0

for med in medications:
    price_per_1000 = ingredient_costs[med]
    reimb_per_cycle = reimbursements[med]

    # Drug cost per fill = (capsules_per_fill / 1000) * price_per_1000 * patients
    # Annual drug cost = drug_cost_per_fill * fills_per_year
    annual_drug_cost_28 = round((caps_28 / 1000.0) * price_per_1000 * patients * fills_28, 2)
    annual_drug_cost_56 = round((caps_56 / 1000.0) * price_per_1000 * patients * fills_56, 2)

    # Determine blister_card_count for this medication
    # 28-day model uses caps_28=56 capsules, 56-day uses caps_56=112
    # Card count matches the capsules per fill
    blister_28 = caps_28
    blister_56 = caps_56
    card_cost_28 = card_costs[blister_28]
    card_cost_56 = card_costs[blister_56]

    # Packaging cost = card_cost * patients * fills
    annual_pkg_28 = round(card_cost_28 * patients * fills_28, 2)
    annual_pkg_56 = round(card_cost_56 * patients * fills_56, 2)

    # Reimbursement: per cycle for 180 patients, so annual = reimb * fills
    annual_reimb_28 = round(reimb_per_cycle * fills_28, 2)
    annual_reimb_56 = round(reimb_per_cycle * fills_56, 2)

    # Margins
    margin_28 = round(annual_reimb_28 - annual_drug_cost_28 - annual_pkg_28, 2)
    margin_56 = round(annual_reimb_56 - annual_drug_cost_56 - annual_pkg_56, 2)
    diff = round(margin_56 - margin_28, 2)

    med_results.append({
        'medication': med,
        'price_per_1000_capsules_usd': price_per_1000,
        'blister_card_count': blister_28,  # will need to check what verifier expects
        'card_cost_usd': card_cost_28,  # will need to check
        'reimbursement_per_cycle_180_patients_usd': reimb_per_cycle,
        'annual_drug_cost_28_day_usd': annual_drug_cost_28,
        'annual_drug_cost_56_day_usd': annual_drug_cost_56,
        'annual_packaging_cost_28_day_usd': annual_pkg_28,
        'annual_packaging_cost_56_day_usd': annual_pkg_56,
        'annual_reimbursement_28_day_usd': annual_reimb_28,
        'annual_reimbursement_56_day_usd': annual_reimb_56,
        'annual_margin_28_day_usd': margin_28,
        'annual_margin_56_day_usd': margin_56,
        'annual_margin_difference_56_minus_28_usd': diff
    })

    total_margin_28 += margin_28
    total_margin_56 += margin_56

total_margin_28 = round(total_margin_28, 2)
total_margin_56 = round(total_margin_56, 2)
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
        'capsules_per_fill_28_day': caps_28,
        'capsules_per_fill_56_day': caps_56,
        'switch_threshold_usd': threshold
    },
    'medications': med_results,
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

# Write summary with comma-formatted currency values (CRITICAL: use :,.2f format)
with open('/root/syncpack_summary.md', 'w') as f:
    f.write('# Syncpack Analysis Summary\n')
    f.write(f'\nTotal 28-day margin (USD): ${total_margin_28:,.2f}\n')
    f.write(f'Total 56-day margin (USD): ${total_margin_56:,.2f}\n')
    f.write(f'Absolute difference (USD): ${abs_diff:,.2f}\n')
    f.write(f'\nFinal decision: {decision}\n')

print('Done. Files written.')
print(f'Total 28-day margin: {total_margin_28:,.2f}')
print(f'Total 56-day margin: {total_margin_56:,.2f}')
print(f'Total difference: {total_diff:,.2f}')
print(f'Absolute difference: {abs_diff:,.2f}')
print(f'Decision: {decision}')
```

3. **BEFORE running solve.py**, inspect the CSV files carefully. After inspecting them, you may need to adjust the script. In particular:
   - Check whether `card_cost.csv` has entries keyed by `blister_card_count` values of 56 and 112, or some other values. If the blister card counts don't match 56/112, you'll need to understand how to match medications to card costs. The `blister_card_count` might be per medication in one of the files.
   - Check whether `reimbursement.csv` has separate rows for 28-day and 56-day cycles, or one row per medication.
   - Adjust the script logic accordingly before running.

4. **Run the script:**
```bash
cd /root && python solve.py
```

5. **Validate the outputs:**
```bash
cat /root/syncpack_analysis.json
cat /root/syncpack_summary.md
```

6. **Check the JSON is valid:**
```bash
python -c "import json; d=json.load(open('/root/syncpack_analysis.json')); print('medications:', len(d['medications'])); print('totals:', d['totals']); print('decision:', d['recommendation']['decision'])"
```

7. **Verify the summary has comma-formatted currency values** (this was the previous failure). Specifically check that values like `-42,908.83` appear with commas as thousands separators. Run:
```bash
grep -oP '\$-?[\d,]+\.\d{2}' /root/syncpack_summary.md
```
Confirm each extracted value has commas in the thousands position.

8. **Run any test file if present:**
```bash
ls /root/test_output.py 2>/dev/null && cd /root && python -m pytest test_output.py -v || echo 'No test file found or tests completed'
```

**CRITICAL NOTES:**
- The previous run failed because currency values in the markdown summary lacked comma thousands separators. Use `f"{value:,.2f}"` formatting for ALL currency values in the summary.
- The JSON schema must have the `recommendation` key at the top level (not `justification` and `decision` at top level). This was a failure mode in a related task.
- Sort medications alphabetically by medication name.
- Round all currency to 2 decimal places.
- After inspecting CSVs, if the data structure differs from assumptions in the script (e.g., blister_card_count is per-medication, or reimbursement has separate 28/56 day entries), adapt the script before running it. The key is to read the actual data first, then code accordingly.

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