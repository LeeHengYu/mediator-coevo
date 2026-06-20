# Task Instruction

Execute the following steps in order:

1. **Inspect the input files** to understand their structure and columns:
 ```
 cat /root/wholesale_price.csv
 cat /root/vial_price.csv
 cat /root/reimbursement.csv
 ```

2. **Write and run a Python script** (`/root/solve.py`) that produces both output files. Use the code below exactly:

```python
import csv
import json

# Read wholesale_price.csv
wholesale = {}
with open('/root/wholesale_price.csv', 'r') as f:
 reader = csv.DictReader(f)
 for row in reader:
 med = row['medication'].strip()
 price = float(row['price_per_1000_tablets_usd'].strip())
 wholesale[med] = price

# Read vial_price.csv
vials = {}
with open('/root/vial_price.csv', 'r') as f:
 reader = csv.DictReader(f)
 for row in reader:
 med = row['medication'].strip()
 vial_size = int(row['vial_size_drams'].strip())
 vial_price = float(row['vial_price_usd'].strip())
 vials[med] = {'vial_size_drams': vial_size, 'vial_price_usd': vial_price}

# Read reimbursement.csv
reimb = {}
with open('/root/reimbursement.csv', 'r') as f:
 reader = csv.DictReader(f)
 for row in reader:
 med = row['medication'].strip()
 reimb_val = float(row['reimbursement_per_fill_300_patients_usd'].strip())
 reimb[med] = reimb_val

# Parameters
patients = 300
fills_90 = 4
fills_100 = 3
tablets_90 = 90
tablets_100 = 100
threshold = 16000

# Use the medication order from wholesale_price.csv (top 10 maintenance meds)
meds_order = []
with open('/root/wholesale_price.csv', 'r') as f:
 reader = csv.DictReader(f)
 for row in reader:
 meds_order.append(row['medication'].strip())

medications = []
for med in meds_order:
 p1000 = wholesale[med]
 vp = vials[med]['vial_price_usd']
 vs = vials[med]['vial_size_drams']
 r = reimb[med]

 # Drug cost = (tablets_per_fill * patients * fills * price_per_1000) / 1000
 drug_cost_90 = round((tablets_90 * patients * fills_90 * p1000) / 1000, 2)
 drug_cost_100 = round((tablets_100 * patients * fills_100 * p1000) / 1000, 2)

 # Supply cost = vial_price * patients * fills
 supply_cost_90 = round(vp * patients * fills_90, 2)
 supply_cost_100 = round(vp * patients * fills_100, 2)

 # Reimbursement = reimbursement_per_fill_300_patients * fills
 reimb_90 = round(r * fills_90, 2)
 reimb_100 = round(r * fills_100, 2)

 # Revenue = reimbursement - drug_cost - supply_cost
 rev_90 = round(reimb_90 - drug_cost_90 - supply_cost_90, 2)
 rev_100 = round(reimb_100 - drug_cost_100 - supply_cost_100, 2)

 diff = round(rev_100 - rev_90, 2)

 medications.append({
 'medication': med,
 'price_per_1000_tablets_usd': p1000,
 'vial_size_drams': vs,
 'vial_price_usd': vp,
 'reimbursement_per_fill_300_patients_usd': r,
 'annual_drug_cost_90_day_usd': drug_cost_90,
 'annual_drug_cost_100_day_usd': drug_cost_100,
 'annual_supply_cost_90_day_usd': supply_cost_90,
 'annual_supply_cost_100_day_usd': supply_cost_100,
 'annual_reimbursement_90_day_usd': reimb_90,
 'annual_reimbursement_100_day_usd': reimb_100,
 'annual_revenue_90_day_usd': rev_90,
 'annual_revenue_100_day_usd': rev_100,
 'annual_revenue_difference_100_minus_90_usd': diff
 })

total_rev_90 = round(sum(m['annual_revenue_90_day_usd'] for m in medications), 2)
total_rev_100 = round(sum(m['annual_revenue_100_day_usd'] for m in medications), 2)
total_diff = round(total_rev_100 - total_rev_90, 2)
abs_diff = round(abs(total_diff), 2)

if abs_diff < threshold:
 decision = 'switch_to_100_day'
 justification = (f'The absolute total revenue difference is ${abs_diff:,.2f}, '
 f'which is below the ${threshold:,.2f} threshold. '
 f'Switching to 100-day fills is recommended.')
else:
 decision = 'keep_90_day'
 justification = (f'The absolute total revenue difference is ${abs_diff:,.2f}, '
 f'which meets or exceeds the ${threshold:,.2f} threshold. '
 f'Keeping 90-day fills is recommended.')

result = {
 'assumptions': {
 'patients_per_medication': patients,
 'fills_per_year_90_day': fills_90,
 'fills_per_year_100_day': fills_100,
 'tablets_per_fill_90_day': tablets_90,
 'tablets_per_fill_100_day': tablets_100,
 'switch_threshold_usd': threshold
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
 json.dump(result, f, indent=2)

print('JSON written successfully.')
print(f'Total 90-day revenue: ${total_rev_90:,.2f}')
print(f'Total 100-day revenue: ${total_rev_100:,.2f}')
print(f'Absolute difference: ${abs_diff:,.2f}')
print(f'Decision: {decision}')

# Write markdown summary with comma-formatted currency values
md = f"""# Refill Policy Analysis Summary

- **Total 90-day annual revenue:** ${total_rev_90:,.2f}
- **Total 100-day annual revenue:** ${total_rev_100:,.2f}
- **Absolute revenue difference:** ${abs_diff:,.2f}
- **Recommendation:** `{decision}`
"""

with open('/root/refill_summary.md', 'w') as f:
 f.write(md)

print('Markdown summary written successfully.')
```

3. **Run the script:**
 ```
 cd /root && python solve.py
 ```

4. **Validate the outputs:**
 - Read `/root/refill_analysis.json` and confirm:
     - The `assumptions` object has all 6 keys with correct values.
     - The `medications` list has 10 entries, each with all 14 required fields.
     - All currency values are rounded to 2 decimal places.
     - `totals` has all 4 keys.
     - `recommendation.decision` is exactly `switch_to_100_day` or `keep_90_day`.
   - Read `/root/refill_summary.md` and confirm:
     - It is 4–8 lines.
     - All dollar amounts use comma-separated thousands format (e.g., `$72,766.88` not `$72766.88`).
     - It contains the exact decision slug (`switch_to_100_day` or `keep_90_day`).

5. **If any validation fails**, fix the issue and re-run. Do not mark the task complete until both files pass all checks.

CRITICAL REMINDERS from previous failure:
- The markdown summary MUST use comma-formatted currency: `f"${value:,.2f}"` — the verifier checks for the comma-separated string.
- Use exact JSON key names from the schema — no abbreviations, no extra keys, no missing keys.

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