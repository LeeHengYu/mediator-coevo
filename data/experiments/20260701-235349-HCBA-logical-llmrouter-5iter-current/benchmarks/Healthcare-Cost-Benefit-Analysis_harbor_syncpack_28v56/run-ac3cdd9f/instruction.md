# Task Instruction

## Task: Healthcare Cost-Benefit Analysis — 28-day vs 56-day Syncpack Comparison

### Step 1: Inspect Input Files
Read and display the contents of all three input CSV files:
```
cat /root/ingredient_cost.csv
cat /root/card_cost.csv
cat /root/reimbursement.csv
```
Understand the columns, medications listed, and how they join together (likely by medication name or blister_card_count).

### Step 2: Write a Python script to perform the analysis

Create `/root/solve.py` with the following logic:

```python
import csv
import json
import math

# Load ingredient_cost.csv
with open('/root/ingredient_cost.csv') as f:
    reader = csv.DictReader(f)
    ingredients = {row['medication'].strip(): row for row in reader}

# Load card_cost.csv
with open('/root/card_cost.csv') as f:
    reader = csv.DictReader(f)
    cards = {}
    for row in reader:
        # Key by blister_card_count (as int)
        count = int(row['blister_card_count'].strip())
        cards[count] = float(row['card_cost_usd'].strip())

# Load reimbursement.csv
with open('/root/reimbursement.csv') as f:
    reader = csv.DictReader(f)
    reimbursements = {row['medication'].strip(): row for row in reader}

# Constants
patients = 180
fills_28 = 12
fills_56 = 6
caps_per_fill_28 = 56
caps_per_fill_56 = 112
threshold = 9000

medications_list = []

# Process each medication (use the medications from ingredient_cost as the master list)
for med_name in sorted(ingredients.keys()):
    ing = ingredients[med_name]
    reimb = reimbursements[med_name]
    
    price_per_1000 = float(ing['price_per_1000_capsules_usd'].strip())
    blister_card_count = int(ing['blister_card_count'].strip()) if 'blister_card_count' in ing else None
    
    # If blister_card_count is not in ingredient_cost, check reimbursement or find it
    # We need to figure out the join. Let me handle both cases:
    # The blister_card_count might be in ingredient_cost.csv or we might need to get it elsewhere
    # We'll check after inspecting the files. For now, try ingredient_cost first.
    
    # If blister_card_count not found in ingredients, try reimbursement
    if blister_card_count is None and 'blister_card_count' in reimb:
        blister_card_count = int(reimb['blister_card_count'].strip())
    
    card_cost = cards[blister_card_count]
    reimb_per_cycle = float(reimb['reimbursement_per_cycle_180_patients_usd'].strip())
    
    # Annual drug cost: (capsules_per_fill * fills_per_year * patients) * (price_per_1000 / 1000)
    annual_drug_cost_28 = round((caps_per_fill_28 * fills_28 * patients) * (price_per_1000 / 1000.0), 2)
    annual_drug_cost_56 = round((caps_per_fill_56 * fills_56 * patients) * (price_per_1000 / 1000.0), 2)
    
    # Annual packaging cost: card_cost * patients * fills_per_year
    annual_pkg_28 = round(card_cost * patients * fills_28, 2)
    annual_pkg_56 = round(card_cost * patients * fills_56, 2)
    
    # Annual reimbursement: reimbursement_per_cycle * fills_per_year
    annual_reimb_28 = round(reimb_per_cycle * fills_28, 2)
    annual_reimb_56 = round(reimb_per_cycle * fills_56, 2)
    
    # Annual margin: reimbursement - drug_cost - packaging_cost
    margin_28 = round(annual_reimb_28 - annual_drug_cost_28 - annual_pkg_28, 2)
    margin_56 = round(annual_reimb_56 - annual_drug_cost_56 - annual_pkg_56, 2)
    
    diff = round(margin_56 - margin_28, 2)
    
    medications_list.append({
        "medication": med_name,
        "price_per_1000_capsules_usd": price_per_1000,
        "blister_card_count": blister_card_count,
        "card_cost_usd": card_cost,
        "reimbursement_per_cycle_180_patients_usd": reimb_per_cycle,
        "annual_drug_cost_28_day_usd": annual_drug_cost_28,
        "annual_drug_cost_56_day_usd": annual_drug_cost_56,
        "annual_packaging_cost_28_day_usd": annual_pkg_28,
        "annual_packaging_cost_56_day_usd": annual_pkg_56,
        "annual_reimbursement_28_day_usd": annual_reimb_28,
        "annual_reimbursement_56_day_usd": annual_reimb_56,
        "annual_margin_28_day_usd": margin_28,
        "annual_margin_56_day_usd": margin_56,
        "annual_margin_difference_56_minus_28_usd": diff
    })

# Totals
total_margin_28 = round(sum(m['annual_margin_28_day_usd'] for m in medications_list), 2)
total_margin_56 = round(sum(m['annual_margin_56_day_usd'] for m in medications_list), 2)
total_diff = round(total_margin_56 - total_margin_28, 2)
abs_diff = round(abs(total_diff), 2)

# Decision
if abs_diff < 9000:
    decision = "convert_to_56_day"
    justification = f"The absolute total margin difference of ${abs_diff:.2f} is below the ${threshold} threshold, so converting to 56-day cycles is recommended."
else:
    decision = "keep_28_day"
    justification = f"The absolute total margin difference of ${abs_diff:.2f} exceeds the ${threshold} threshold, so keeping 28-day cycles is recommended."

result = {
    "assumptions": {
        "patients_per_medication": patients,
        "fills_per_year_28_day": fills_28,
        "fills_per_year_56_day": fills_56,
        "capsules_per_fill_28_day": caps_per_fill_28,
        "capsules_per_fill_56_day": caps_per_fill_56,
        "switch_threshold_usd": threshold
    },
    "medications": medications_list,
    "totals": {
        "total_annual_margin_28_day_usd": total_margin_28,
        "total_annual_margin_56_day_usd": total_margin_56,
        "total_annual_margin_difference_56_minus_28_usd": total_diff,
        "absolute_total_margin_difference_usd": abs_diff
    },
    "recommendation": {
        "decision": decision,
        "justification": justification
    }
}

# Write JSON
with open('/root/syncpack_analysis.json', 'w') as f:
    json.dump(result, f, indent=2)

print("JSON written.")
print(json.dumps(result, indent=2))

# Write summary markdown
with open('/root/syncpack_summary.md', 'w') as f:
    f.write(f"# Syncpack Analysis Summary\n")
    f.write(f"\n")
    f.write(f"Total 28-day annual margin: ${total_margin_28:,.2f} USD\n")
    f.write(f"Total 56-day annual margin: ${total_margin_56:,.2f} USD\n")
    f.write(f"Absolute margin difference: ${abs_diff:,.2f} USD\n")
    f.write(f"\n")
    f.write(f"**Decision: {decision}**\n")
    f.write(f"\n")
    f.write(f"{justification}\n")

print("Summary written.")
```

**IMPORTANT:** After inspecting the CSV files in Step 1, you MUST adapt the script to match the actual column names and structure of the CSV files. The column names I used above are guesses based on the task description. Key things to verify:
- What columns are in each CSV? Especially check if `blister_card_count` is in `ingredient_cost.csv` or elsewhere.
- How do the CSVs join together? By medication name? By blister_card_count?
- What is the exact column name for reimbursement per cycle?
- Are there any extra whitespace or formatting issues?

### Step 3: Run the script
```
cd /root && python3 solve.py
```

### Step 4: Validate the outputs

1. Verify JSON is valid:
```
python3 -c "import json; d=json.load(open('/root/syncpack_analysis.json')); print('Keys:', list(d.keys())); print('Meds count:', len(d['medications'])); print('Totals:', d['totals']); print('Decision:', d['recommendation']['decision'])"
```

2. Verify medications are sorted alphabetically:
```
python3 -c "import json; d=json.load(open('/root/syncpack_analysis.json')); names=[m['medication'] for m in d['medications']]; print('Sorted:', names == sorted(names)); print(names)"
```

3. Verify all currency values are rounded to 2 decimals (spot check):
```
python3 -c "
import json
d=json.load(open('/root/syncpack_analysis.json'))
for m in d['medications']:
    for k,v in m.items():
        if isinstance(v, float):
            s = str(v)
            if '.' in s and len(s.split('.')[1]) > 2:
                print(f'ROUNDING ISSUE: {m[\"medication\"]} {k} = {v}')
print('Rounding check done')
"
```

4. Verify the summary markdown:
```
cat /root/syncpack_summary.md
```
Confirm it has 4-8 non-empty lines, includes total 28-day margin, total 56-day margin, absolute difference, and the exact decision slug (`convert_to_56_day` or `keep_28_day`).

5. Verify the annual drug costs are identical for 28-day and 56-day (since 56*12 = 112*6 = 672 capsules/year/patient, the annual drug cost should be the same for both models). This is a sanity check.
```
python3 -c "import json; d=json.load(open('/root/syncpack_analysis.json')); [print(f'{m[\"medication\"]}: 28={m[\"annual_drug_cost_28_day_usd\"]}, 56={m[\"annual_drug_cost_56_day_usd\"]}') for m in d['medications']]"
```

6. Verify the margin difference is driven purely by packaging cost differences (since drug cost and reimbursement cycles differ).

### Key Notes
- The drug cost should be the same for both models since total capsules/year = 672 per patient in both cases (56×12 = 112×6 = 672).
- The packaging cost differs: 28-day has 12 fills × card_cost × 180 patients; 56-day has 6 fills × card_cost × 180 patients.
- The reimbursement differs: 28-day has 12 cycles × reimb_per_cycle; 56-day has 6 cycles × reimb_per_cycle.
- The margin difference per medication = (reimb_per_cycle × 6 - drug_cost_56 - pkg_56) - (reimb_per_cycle × 12 - drug_cost_28 - pkg_28) = -6 × reimb_per_cycle + card_cost × 180 × 6 (since drug costs cancel).
- Make sure to handle the `card_cost.csv` lookup correctly — it maps blister_card_count to card_cost_usd. Each medication has a blister_card_count that determines which card cost applies.

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