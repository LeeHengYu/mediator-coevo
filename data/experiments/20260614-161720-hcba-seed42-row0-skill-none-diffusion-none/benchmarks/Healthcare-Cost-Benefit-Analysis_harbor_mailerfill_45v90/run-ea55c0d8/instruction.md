# Task Instruction

Execute the following steps in order:

## Step 1: Inspect all input files
```bash
cat /root/compound_cost.csv
cat /root/mailer_cost.csv
cat /root/base_payment.csv
cat /root/service_fee.csv
```

## Step 2: Create a Python script to compute everything and produce both output files

Create `/root/solve.py` with the following logic:

```python
import csv
import json
import locale

# Read CSVs
def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))

compound = read_csv('/root/compound_cost.csv')
mailer = read_csv('/root/mailer_cost.csv')
base_pay = read_csv('/root/base_payment.csv')
service = read_csv('/root/service_fee.csv')

# Build lookup dicts
# compound_cost.csv should have columns: medication, price_per_1000_doses_usd, mailer_format (or similar)
# Inspect columns carefully and adapt

# Print column headers for debugging
print('compound cols:', list(compound[0].keys()) if compound else 'EMPTY')
print('mailer cols:', list(mailer[0].keys()) if mailer else 'EMPTY')
print('base_pay cols:', list(base_pay[0].keys()) if base_pay else 'EMPTY')
print('service cols:', list(service[0].keys()) if service else 'EMPTY')

print('\ncompound rows:')
for r in compound: print(r)
print('\nmailer rows:')
for r in mailer: print(r)
print('\nbase_pay rows:')
for r in base_pay: print(r)
print('\nservice rows:')
for r in service: print(r)
```

Run it first to see the actual column names and data, then proceed.

## Step 3: After inspecting, write the full solution script

Create `/root/solve.py` (overwrite) that:

1. Reads all four CSVs.
2. Builds lookup dicts:
   - `mailer_cost_lookup[mailer_format]` -> `mailer_cost_usd` (float)
   - `base_payment_lookup[medication]` -> `base_payment_per_fill_150_patients_usd` (float)
   - `service_fee_lookup[medication]` -> `service_fee_per_fill_150_patients_usd` (float)
3. For each row in `compound_cost.csv` (which should have `medication`, `price_per_1000_doses_usd`, and `mailer_format`):
   - `price_per_1000 = float(row['price_per_1000_doses_usd'])`
   - `mailer_fmt = row['mailer_format']`
   - `mailer_cost = mailer_cost_lookup[mailer_fmt]`
   - `base_payment = base_payment_lookup[medication]`
   - `service_fee = service_fee_lookup[medication]`
   - `total_payment_per_fill = base_payment + service_fee`
   - **Drug cost calculation**: Both models have same annual doses: `150 patients * 1 dose/day * 360 days` — BUT WAIT. The fills model is 45*8=360 and 90*4=360 doses per patient per year. So annual doses per medication = 150 * 360 = 54,000. Annual drug cost = `54000 / 1000 * price_per_1000` for BOTH models (identical).
   - `annual_drug_cost_45 = round(150 * 45 * 8 / 1000 * price_per_1000, 2)`
   - `annual_drug_cost_90 = round(150 * 90 * 4 / 1000 * price_per_1000, 2)`
   - `annual_mailer_cost_45 = round(150 * 8 * mailer_cost, 2)`
   - `annual_mailer_cost_90 = round(150 * 4 * mailer_cost, 2)`
   - `annual_payment_45 = round(8 * total_payment_per_fill, 2)`
   - `annual_payment_90 = round(4 * total_payment_per_fill, 2)`
   - `annual_margin_45 = round(annual_payment_45 - annual_drug_cost_45 - annual_mailer_cost_45, 2)`
   - `annual_margin_90 = round(annual_payment_90 - annual_drug_cost_90 - annual_mailer_cost_90, 2)`
   - `difference = round(annual_margin_90 - annual_margin_45, 2)`
4. Sort medications alphabetically by `medication` name.
5. Compute totals by summing all per-medication values.
6. Decision: if `abs(total_difference) < 8500` then `shift_to_90_day`, else `keep_45_day`.
7. Write `/root/mailer_policy_analysis.json` with the EXACT schema from the instructions. Use exactly these keys — no extra keys, no missing keys:
   - `assumptions` with exactly: `patients_per_medication`, `fills_per_year_45_day`, `fills_per_year_90_day`, `doses_per_fill_45_day`, `doses_per_fill_90_day`, `switch_threshold_usd`
   - `medications` array with exactly the fields listed in the schema
   - `totals` with exactly: `total_annual_margin_45_day_usd`, `total_annual_margin_90_day_usd`, `total_annual_margin_difference_90_minus_45_usd`, `absolute_total_margin_difference_usd`
   - `recommendation` with `decision` and `justification`
8. Write `/root/mailer_policy_summary.md` with 4-8 non-empty lines containing:
   - Total 45-day margin formatted with commas (e.g., `27,000.00`)
   - Total 90-day margin formatted with commas
   - Absolute difference formatted with commas
   - The exact decision slug (`shift_to_90_day` or `keep_45_day`)
   - Use Python's `f"{value:,.2f}"` for comma formatting

## Step 4: Run the script
```bash
python3 /root/solve.py
```

## Step 5: Validate outputs
```bash
cat /root/mailer_policy_analysis.json | python3 -c "import sys,json; d=json.load(sys.stdin); print('Keys:', list(d.keys())); print('Assumptions keys:', list(d['assumptions'].keys())); print('Med keys:', list(d['medications'][0].keys()) if d['medications'] else 'EMPTY'); print('Totals keys:', list(d['totals'].keys())); print('Rec:', d['recommendation']); print('Num meds:', len(d['medications'])); print('Sorted check:', [m['medication'] for m in d['medications']])"
cat /root/mailer_policy_summary.md
```

Verify:
- JSON has exactly 4 top-level keys: assumptions, medications, totals, recommendation
- assumptions has exactly 6 keys as specified
- Each medication object has exactly 16 keys as specified
- totals has exactly 4 keys as specified
- Medications are sorted alphabetically
- Summary has 4-8 non-empty lines with comma-formatted currency values
- Drug costs are identical for both models (since 45*8 = 90*4 = 360 doses/patient/year)

## CRITICAL NOTES from previous failures:
- Do NOT add extra keys like `dose_per_patient_per_day` or `effective_days_per_year` to assumptions
- Do NOT omit `doses_per_fill_45_day` or `doses_per_fill_90_day` from assumptions
- Format currency in the markdown summary WITH commas: use `f"{val:,.2f}"` to get `27,000.00` not `27000.00`
- Match the JSON schema EXACTLY — no extra fields, no missing fields, no renamed fields

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
Task metadata: author_email=gpt54@example.com, author_name=GPT-5.4, category=financial-analysis, difficulty=medium, tags=[mailer-program, csv, json, revenue-merge, decision-analysis].
Verifier config: timeout_sec=900.0.