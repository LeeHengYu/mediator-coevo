# Task Instruction

Execute the following steps in order:

## Step 1: Inspect the input CSV files

Read and display the contents of:
- `/root/wholesale_price.csv`
- `/root/vial_price.csv`
- `/root/reimbursement.csv`

Note the exact column names, medication names (spelling, casing), and numeric values.

## Step 2: Optionally inspect the PDFs

If any CSV seems incomplete or ambiguous, glance at `/root/Wholesale_Price.pdf` and `/root/Reimbursement.pdf` for clarification.

## Step 3: Write a Python script `/root/solve.py` that does the following

```python
import csv, json, math

# 1. Read wholesale_price.csv -> dict keyed by medication name
#    Expected columns include medication name and price_per_1000_tablets_usd
# 2. Read vial_price.csv -> dict keyed by medication name
#    Expected columns include medication name, vial_size_drams, vial_price_usd
# 3. Read reimbursement.csv -> dict keyed by medication name
#    Expected column for reimbursement per fill for 300 patients

# Constants
patients = 300
fills_90 = 4
fills_100 = 3
tablets_90 = 90
tablets_100 = 100
threshold = 16000

# For each medication (preserve the order from wholesale_price.csv or whichever
# CSV lists all 10; use the order they appear in the CSV):
#
# annual_drug_cost_90  = (price_per_1000_tablets / 1000) * tablets_90 * patients * fills_90
# annual_drug_cost_100 = (price_per_1000_tablets / 1000) * tablets_100 * patients * fills_100
#
# annual_supply_cost_90  = vial_price_usd * patients * fills_90
# annual_supply_cost_100 = vial_price_usd * patients * fills_100
#
# annual_reimbursement_90  = reimbursement_per_fill_300_patients * fills_90
# annual_reimbursement_100 = reimbursement_per_fill_300_patients * fills_100
#
# annual_revenue_90  = annual_reimbursement_90  - annual_drug_cost_90  - annual_supply_cost_90
# annual_revenue_100 = annual_reimbursement_100 - annual_drug_cost_100 - annual_supply_cost_100
#
# difference = annual_revenue_100 - annual_revenue_90
#
# Round ALL currency values to 2 decimal places.

# Build the medications list, totals, recommendation per the schema.
# Decision rule:
#   if abs(total_difference) < 16000 -> "switch_to_100_day"
#   else -> "keep_90_day"

# Write /root/refill_analysis.json with indent=2
# Write /root/refill_summary.md (4-8 lines) including:
#   - total 90-day revenue
#   - total 100-day revenue  
#   - absolute difference
#   - exact slug: switch_to_100_day or keep_90_day
```

IMPORTANT details for the script:
- Use the EXACT medication names as they appear in the CSV files (preserve spelling and casing).
- The JSON output must use EXACTLY these top-level keys: `assumptions`, `medications`, `totals`, `recommendation`.
- Each medication dict must have EXACTLY these keys (no extras, no missing):
  `medication`, `price_per_1000_tablets_usd`, `vial_size_drams`, `vial_price_usd`, `reimbursement_per_fill_300_patients_usd`, `annual_drug_cost_90_day_usd`, `annual_drug_cost_100_day_usd`, `annual_supply_cost_90_day_usd`, `annual_supply_cost_100_day_usd`, `annual_reimbursement_90_day_usd`, `annual_reimbursement_100_day_usd`, `annual_revenue_90_day_usd`, `annual_revenue_100_day_usd`, `annual_revenue_difference_100_minus_90_usd`
- The `totals` dict must have EXACTLY: `total_annual_revenue_90_day_usd`, `total_annual_revenue_100_day_usd`, `total_annual_revenue_difference_100_minus_90_usd`, `absolute_total_revenue_difference_usd`
- The `recommendation` dict must have `decision` (one of the two exact slugs) and `justification` (a brief string).
- All numeric currency values must be `float` rounded to 2 decimals (use `round(x, 2)`).
- `vial_size_drams` should be `int`.
- Write JSON with `json.dump(..., indent=2)`.

## Step 4: Run the script

```bash
cd /root && python solve.py
```

## Step 5: Validate the outputs

1. `cat /root/refill_analysis.json` — verify it parses, has all required keys, correct field names, 10 medications, and all values are rounded to 2 decimals.
2. `cat /root/refill_summary.md` — verify it has 4-8 lines and includes the required information with the exact decision slug.
3. Run `python -c "import json; d=json.load(open('/root/refill_analysis.json')); assert 'assumptions' in d; assert 'medications' in d; assert 'totals' in d; assert 'recommendation' in d; assert len(d['medications'])==10; print('Schema OK')"` to confirm basic schema compliance.

## Step 6: If any validation fails, fix and re-run

Do not mark complete until both output files exist and pass validation.

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