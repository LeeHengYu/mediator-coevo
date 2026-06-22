# Task Instruction

## Task: Retail Pharmacy Auto-Refill Cost-Benefit Analysis

You must analyze whether a pharmacy should keep 90-day fills or switch to 100-day fills for its top 10 maintenance medications, then produce two output files.

### Step 1: Inspect Input Files

Read and display the contents of all three input CSV files:
```
cat /root/wholesale_price.csv
cat /root/vial_price.csv
cat /root/reimbursement.csv
```

Understand the column names and data types. Also optionally glance at the PDFs if any CSV seems incomplete or ambiguous.

### Step 2: Write a Python Script to Perform the Analysis

Create a Python script `/root/analyze.py` that:

1. **Loads the three CSVs** using pandas.
2. **Merges them** on medication name (be careful about exact column names and string matching — strip whitespace, normalize case if needed).
3. **For each of the 10 medications, computes:**

   **Drug cost per fill** = `(tablets_per_fill / 1000) * price_per_1000_tablets_usd * 300` (for 300 patients)
   
   - `annual_drug_cost_90_day = (90 / 1000) * price_per_1000_tablets_usd * 300 * 4`
   - `annual_drug_cost_100_day = (100 / 1000) * price_per_1000_tablets_usd * 300 * 3`

   **Supply cost per fill** = `vial_price_usd * 300` (one vial per patient per fill)
   
   - `annual_supply_cost_90_day = vial_price_usd * 300 * 4`
   - `annual_supply_cost_100_day = vial_price_usd * 300 * 3`

   **Reimbursement**: The `reimbursement.csv` gives `reimbursement_per_fill_300_patients_usd` — this is the total reimbursement for one fill for all 300 patients.
   
   - `annual_reimbursement_90_day = reimbursement_per_fill_300_patients_usd * 4`
   - `annual_reimbursement_100_day = reimbursement_per_fill_300_patients_usd * 3`

   **Revenue**:
   - `annual_revenue_90_day = annual_reimbursement_90_day - annual_drug_cost_90_day - annual_supply_cost_90_day`
   - `annual_revenue_100_day = annual_reimbursement_100_day - annual_drug_cost_100_day - annual_supply_cost_100_day`
   - `annual_revenue_difference_100_minus_90 = annual_revenue_100_day - annual_revenue_90_day`

4. **Computes totals:**
   - `total_annual_revenue_90_day = sum of all annual_revenue_90_day`
   - `total_annual_revenue_100_day = sum of all annual_revenue_100_day`
   - `total_annual_revenue_difference = total_annual_revenue_100_day - total_annual_revenue_90_day`
   - `absolute_total_revenue_difference = abs(total_annual_revenue_difference)`

5. **Decision rule:**
   - If `absolute_total_revenue_difference < 16000`, decision = `"switch_to_100_day"`
   - Otherwise, decision = `"keep_90_day"`

6. **Round ALL currency values to 2 decimal places.**

7. **Writes `/root/refill_analysis.json`** with the EXACT schema specified below. The `medications` array must have one object per medication. Field names must match exactly:
   ```
   assumptions.patients_per_medication: 300
   assumptions.fills_per_year_90_day: 4
   assumptions.fills_per_year_100_day: 3
   assumptions.tablets_per_fill_90_day: 90
   assumptions.tablets_per_fill_100_day: 100
   assumptions.switch_threshold_usd: 16000
   ```
   Each medication object must have these exact keys:
   - `medication` (string)
   - `price_per_1000_tablets_usd` (float)
   - `vial_size_drams` (int)
   - `vial_price_usd` (float)
   - `reimbursement_per_fill_300_patients_usd` (float)
   - `annual_drug_cost_90_day_usd` (float)
   - `annual_drug_cost_100_day_usd` (float)
   - `annual_supply_cost_90_day_usd` (float)
   - `annual_supply_cost_100_day_usd` (float)
   - `annual_reimbursement_90_day_usd` (float)
   - `annual_reimbursement_100_day_usd` (float)
   - `annual_revenue_90_day_usd` (float)
   - `annual_revenue_100_day_usd` (float)
   - `annual_revenue_difference_100_minus_90_usd` (float)

   Totals object keys:
   - `total_annual_revenue_90_day_usd`
   - `total_annual_revenue_100_day_usd`
   - `total_annual_revenue_difference_100_minus_90_usd`
   - `absolute_total_revenue_difference_usd`

   Recommendation:
   - `decision`: exactly `"switch_to_100_day"` or `"keep_90_day"`
   - `justification`: a brief string explaining the decision referencing the threshold and the absolute difference.

8. **Writes `/root/refill_summary.md`** — 4 to 8 lines, must include:
   - Total 90-day revenue in USD
   - Total 100-day revenue in USD
   - Absolute difference in USD
   - The exact decision slug: `switch_to_100_day` or `keep_90_day`

### Step 3: Run the Script
```
python3 /root/analyze.py
```

### Step 4: Validate Outputs

1. Display the JSON output:
```
cat /root/refill_analysis.json
```
Verify:
- The `assumptions` block has the correct integer/float values.
- There are exactly 10 medication entries.
- All currency fields are rounded to 2 decimal places (no more, no fewer).
- The `totals` values equal the sum of individual medication values.
- The `decision` field is one of the two exact slugs.
- The JSON is valid (parseable).

2. Display the summary:
```
cat /root/refill_summary.md
```
Verify it is 4-8 lines and contains all four required pieces of information with the exact slug.

3. Quick sanity check: Run `python3 -c "import json; d=json.load(open('/root/refill_analysis.json')); print(len(d['medications']), d['totals'], d['recommendation']['decision'])"` to confirm parseability and key structure.

### Important Notes
- When merging CSVs, be very careful about medication name matching. Strip whitespace, and check if names match exactly across files. If they don't match perfectly, print the mismatches and handle them.
- The `vial_price.csv` may contain vial size info — include `vial_size_drams` in each medication entry.
- The reimbursement CSV column may be named differently — inspect it carefully and use the actual column name.
- All float values in the JSON must have exactly 2 decimal places when they represent USD amounts. Use `round(value, 2)` in Python.
- Do NOT invent data. Use only what is in the CSV files.

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