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

Create a Python script `/root/analyze.py` that does the following:

1. **Load the three CSVs** using pandas.
2. **Merge them** on medication name. Be careful about column name matching — inspect the actual column headers. The medication name column may differ across files (e.g., `medication`, `drug_name`, etc.). Normalize/strip whitespace as needed.
3. **For each of the 10 medications, compute:**

   - `annual_drug_cost_90_day = (price_per_1000_tablets / 1000) * 90 * 4 * 300`
   - `annual_drug_cost_100_day = (price_per_1000_tablets / 1000) * 100 * 3 * 300`
   - `annual_supply_cost_90_day = vial_price * 300 * 4`  (one vial per patient per fill)
   - `annual_supply_cost_100_day = vial_price * 300 * 3`
   - `annual_reimbursement_90_day = reimbursement_per_fill_300_patients * 4`
   - `annual_reimbursement_100_day = reimbursement_per_fill_300_patients * 3`
   - `annual_revenue_90_day = annual_reimbursement_90_day - annual_drug_cost_90_day - annual_supply_cost_90_day`
   - `annual_revenue_100_day = annual_reimbursement_100_day - annual_drug_cost_100_day - annual_supply_cost_100_day`
   - `annual_revenue_difference = annual_revenue_100_day - annual_revenue_90_day`

4. **Round all currency values to 2 decimal places.**

5. **Compute totals:**
   - `total_annual_revenue_90_day` = sum of all medications' `annual_revenue_90_day`
   - `total_annual_revenue_100_day` = sum of all medications' `annual_revenue_100_day`
   - `total_annual_revenue_difference` = sum of all medications' `annual_revenue_difference`
   - `absolute_total_revenue_difference` = abs(total_annual_revenue_difference)

6. **Decision rule:**
   - If `absolute_total_revenue_difference < 16000`, decision = `"switch_to_100_day"`
   - Otherwise, decision = `"keep_90_day"`

7. **Write `/root/refill_analysis.json`** with the exact schema specified:
   - `assumptions` object with the fixed values
   - `medications` array with one object per medication containing all fields listed in the schema
   - `totals` object
   - `recommendation` object with `decision` (the exact slug) and `justification` (a brief sentence)
   - Use `json.dumps(..., indent=2)` for formatting
   - Ensure all float values are rounded to 2 decimals

8. **Write `/root/refill_summary.md`** with 4-8 lines that includes:
   - Total 90-day revenue (USD)
   - Total 100-day revenue (USD)
   - Absolute difference (USD)
   - Final decision using the exact slug `switch_to_100_day` or `keep_90_day`

### Step 3: Run the Script
```
python3 /root/analyze.py
```

### Step 4: Validate Outputs

1. `cat /root/refill_analysis.json` — verify it parses as valid JSON, has exactly 10 medications, all required fields present, all values rounded to 2 decimals, decision slug is exactly one of `switch_to_100_day` or `keep_90_day`.
2. `cat /root/refill_summary.md` — verify it has 4-8 lines, includes the total revenues, absolute difference, and the exact decision slug.
3. Run `python3 -c "import json; d=json.load(open('/root/refill_analysis.json')); print(len(d['medications']), d['totals'], d['recommendation'])"` to confirm structure.

### Important Notes
- The medication names in the JSON output should match exactly what appears in the source data.
- The `vial_size_drams` field should come from the vial_price.csv if available.
- Do NOT invent data. All values must come from the CSV files.
- All currency values must be rounded to exactly 2 decimal places (use Python's `round(x, 2)`).
- The `reimbursement_per_fill_300_patients_usd` is the value directly from reimbursement.csv — it already represents the reimbursement for 300 patients per fill.
- Pay close attention to the drug cost formula: it's price_per_1000_tablets / 1000 to get per-tablet cost, then multiply by tablets_per_fill * fills_per_year * 300 patients.

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