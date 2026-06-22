# Task Instruction

Execute the following steps in order:

## Step 1: Inspect input files
Read and display the contents of:
- `/root/wholesale_price.csv`
- `/root/vial_price.csv`
- `/root/reimbursement.csv`

Also check if the PDFs exist (just confirm presence, no need to parse them since CSVs are the machine-readable inputs).

## Step 2: Write and run a Python script

Create and execute a Python script `/root/solve.py` that does the following:

```python
import csv
import json
import os

# 1. Read wholesale_price.csv
# Expected columns include medication name and price_per_1000_tablets_usd
wholesale = {}
with open('/root/wholesale_price.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Print column names on first iteration to debug
        med = row.get('medication') or row.get('Medication') or row.get('drug') or row.get('Drug')
        price_key = [k for k in row.keys() if '1000' in k.lower() or 'price_per' in k.lower()]
        if not med:
            # Try first column
            med = list(row.values())[0]
        price = float(row[price_key[0]]) if price_key else float(list(row.values())[1])
        wholesale[med.strip()] = price

# 2. Read vial_price.csv
# Expected columns: medication, vial_size_drams, vial_price_usd
vial_info = {}
with open('/root/vial_price.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        med = None
        for k in row.keys():
            if 'med' in k.lower() or 'drug' in k.lower():
                med = row[k].strip()
                break
        if not med:
            med = list(row.values())[0].strip()
        
        vial_size = None
        vial_price = None
        for k in row.keys():
            kl = k.lower()
            if 'size' in kl or 'dram' in kl:
                vial_size = row[k]
            if 'price' in kl or 'cost' in kl:
                vial_price = row[k]
        
        if vial_size is None:
            vial_size = list(row.values())[1]
        if vial_price is None:
            vial_price = list(row.values())[2]
            
        vial_info[med] = {
            'vial_size_drams': int(float(vial_size)),
            'vial_price_usd': float(vial_price)
        }

# 3. Read reimbursement.csv
# Expected: medication, reimbursement_per_fill_300_patients_usd
reimbursement = {}
with open('/root/reimbursement.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        med = None
        for k in row.keys():
            if 'med' in k.lower() or 'drug' in k.lower():
                med = row[k].strip()
                break
        if not med:
            med = list(row.values())[0].strip()
        
        reimb = None
        for k in row.keys():
            if 'reimb' in k.lower():
                reimb = row[k]
                break
        if reimb is None:
            reimb = list(row.values())[1]
        
        reimbursement[med] = float(reimb)

print("Wholesale:", wholesale)
print("Vial info:", vial_info)
print("Reimbursement:", reimbursement)

# Parameters
patients = 300
fills_90 = 4
fills_100 = 3
tablets_90 = 90
tablets_100 = 100
threshold = 16000

# Build medication list - use the order from wholesale_price.csv as canonical
med_names = list(wholesale.keys())

medications = []
for med in med_names:
    ppk = wholesale[med]  # price per 1000 tablets
    vi = vial_info[med]
    vp = vi['vial_price_usd']
    vs = vi['vial_size_drams']
    reimb_per_fill = reimbursement[med]
    
    # Drug cost = (tablets_per_fill * patients * fills * price_per_1000) / 1000
    annual_drug_cost_90 = round((tablets_90 * patients * fills_90 * ppk) / 1000, 2)
    annual_drug_cost_100 = round((tablets_100 * patients * fills_100 * ppk) / 1000, 2)
    
    # Supply cost = vial_price * patients * fills
    annual_supply_cost_90 = round(vp * patients * fills_90, 2)
    annual_supply_cost_100 = round(vp * patients * fills_100, 2)
    
    # Reimbursement: the CSV gives reimbursement per fill for 300 patients
    # So annual = reimb_per_fill * fills
    annual_reimb_90 = round(reimb_per_fill * fills_90, 2)
    annual_reimb_100 = round(reimb_per_fill * fills_100, 2)
    
    # Revenue = reimbursement - drug_cost - supply_cost
    annual_rev_90 = round(annual_reimb_90 - annual_drug_cost_90 - annual_supply_cost_90, 2)
    annual_rev_100 = round(annual_reimb_100 - annual_drug_cost_100 - annual_supply_cost_100, 2)
    
    diff = round(annual_rev_100 - annual_rev_90, 2)
    
    medications.append({
        "medication": med,
        "price_per_1000_tablets_usd": ppk,
        "vial_size_drams": vs,
        "vial_price_usd": vp,
        "reimbursement_per_fill_300_patients_usd": reimb_per_fill,
        "annual_drug_cost_90_day_usd": annual_drug_cost_90,
        "annual_drug_cost_100_day_usd": annual_drug_cost_100,
        "annual_supply_cost_90_day_usd": annual_supply_cost_90,
        "annual_supply_cost_100_day_usd": annual_supply_cost_100,
        "annual_reimbursement_90_day_usd": annual_reimb_90,
        "annual_reimbursement_100_day_usd": annual_reimb_100,
        "annual_revenue_90_day_usd": annual_rev_90,
        "annual_revenue_100_day_usd": annual_rev_100,
        "annual_revenue_difference_100_minus_90_usd": diff
    })

total_rev_90 = round(sum(m['annual_revenue_90_day_usd'] for m in medications), 2)
total_rev_100 = round(sum(m['annual_revenue_100_day_usd'] for m in medications), 2)
total_diff = round(total_rev_100 - total_rev_90, 2)
abs_diff = round(abs(total_diff), 2)

if abs_diff < 16000:
    decision = "switch_to_100_day"
    justification = f"The absolute total revenue difference is ${abs_diff:,.2f}, which is below the ${threshold:,} threshold. Switching to 100-day fills is recommended as the financial impact is minimal."
else:
    decision = "keep_90_day"
    justification = f"The absolute total revenue difference is ${abs_diff:,.2f}, which exceeds the ${threshold:,} threshold. Keeping 90-day fills is recommended to avoid significant revenue impact."

result = {
    "assumptions": {
        "patients_per_medication": patients,
        "fills_per_year_90_day": fills_90,
        "fills_per_year_100_day": fills_100,
        "tablets_per_fill_90_day": tablets_90,
        "tablets_per_fill_100_day": tablets_100,
        "switch_threshold_usd": threshold
    },
    "medications": medications,
    "totals": {
        "total_annual_revenue_90_day_usd": total_rev_90,
        "total_annual_revenue_100_day_usd": total_rev_100,
        "total_annual_revenue_difference_100_minus_90_usd": total_diff,
        "absolute_total_revenue_difference_usd": abs_diff
    },
    "recommendation": {
        "decision": decision,
        "justification": justification
    }
}

with open('/root/refill_analysis.json', 'w') as f:
    json.dump(result, f, indent=2)

print("\nJSON written to /root/refill_analysis.json")
print(json.dumps(result, indent=2))

# Write summary
with open('/root/refill_summary.md', 'w') as f:
    f.write(f"# Refill Policy Analysis Summary\n\n")
    f.write(f"- **Total 90-day annual revenue:** ${total_rev_90:,.2f} USD\n")
    f.write(f"- **Total 100-day annual revenue:** ${total_rev_100:,.2f} USD\n")
    f.write(f"- **Absolute revenue difference:** ${abs_diff:,.2f} USD\n")
    f.write(f"- **Recommendation:** {decision}\n")

print("\nSummary written to /root/refill_summary.md")
```

IMPORTANT: Before running the script, first display the raw CSV files so you can see the exact column names and data. Then adjust the script column-name parsing if needed to match the actual headers.

## Step 3: Validate outputs

1. Read `/root/refill_analysis.json` and verify:
   - It has exactly the keys: `assumptions`, `medications`, `totals`, `recommendation`
   - `medications` is a list of 10 items (top 10 maintenance medications)
   - All currency values are rounded to 2 decimal places
   - The `decision` field is exactly one of `switch_to_100_day` or `keep_90_day`
   - `absolute_total_revenue_difference_usd` equals `abs(total_annual_revenue_difference_100_minus_90_usd)`
   - The decision rule is correctly applied: if abs_diff < 16000 then switch_to_100_day, else keep_90_day

2. Read `/root/refill_summary.md` and verify:
   - It is 4-8 lines
   - Contains total 90-day revenue in USD
   - Contains total 100-day revenue in USD  
   - Contains absolute difference in USD
   - Contains the exact decision slug (`switch_to_100_day` or `keep_90_day`)

3. Spot-check one medication's calculations manually:
   - annual_drug_cost_90 = (90 * 300 * 4 * price_per_1000) / 1000
   - annual_drug_cost_100 = (100 * 300 * 3 * price_per_1000) / 1000
   - annual_supply_cost_90 = vial_price * 300 * 4
   - annual_supply_cost_100 = vial_price * 300 * 3
   - annual_reimbursement_90 = reimb_per_fill * 4
   - annual_reimbursement_100 = reimb_per_fill * 3
   - annual_revenue = reimbursement - drug_cost - supply_cost

If any validation fails, fix the issue and re-run.

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