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

Understand the column names and data types. Also optionally glance at the PDFs to cross-check if needed (but the CSVs are the authoritative machine-readable inputs).

### Step 2: Write a Python Script to Perform the Analysis

Create and run a Python script `/root/analyze.py` that does the following:

1. **Load all three CSVs** using pandas. Join/merge them on the medication name column. Ensure all 10 medications are present after merging. Print the merged dataframe to verify.

2. **For each medication, compute** (all values rounded to 2 decimals at the end):

   - `annual_drug_cost_90_day = (price_per_1000_tablets / 1000) * 90 * 4 * 300`
   - `annual_drug_cost_100_day = (price_per_1000_tablets / 1000) * 100 * 3 * 300`
   - `annual_supply_cost_90_day = vial_price * 300 * 4`  (one vial per patient per fill)
   - `annual_supply_cost_100_day = vial_price * 300 * 3`
   - `annual_reimbursement_90_day = reimbursement_per_fill_300_patients * 4`
   - `annual_reimbursement_100_day = reimbursement_per_fill_300_patients * 3`
   - `annual_revenue_90_day = annual_reimbursement_90_day - annual_drug_cost_90_day - annual_supply_cost_90_day`
   - `annual_revenue_100_day = annual_reimbursement_100_day - annual_drug_cost_100_day - annual_supply_cost_100_day`
   - `annual_revenue_difference_100_minus_90 = annual_revenue_100_day - annual_revenue_90_day`

3. **Compute totals**:
   - `total_annual_revenue_90_day` = sum of all medications' `annual_revenue_90_day`
   - `total_annual_revenue_100_day` = sum of all medications' `annual_revenue_100_day`
   - `total_annual_revenue_difference_100_minus_90` = sum of all medications' `annual_revenue_difference_100_minus_90`
   - `absolute_total_revenue_difference` = abs(total_annual_revenue_difference_100_minus_90)

4. **Decision rule**:
   - If `absolute_total_revenue_difference < 16000`, decision = `"switch_to_100_day"`
   - Otherwise, decision = `"keep_90_day"`

5. **Round ALL currency values to 2 decimal places** before writing output.

6. **Write `/root/refill_analysis.json`** with EXACTLY the schema specified:
   - `assumptions` object with the fixed parameters
   - `medications` array of 10 objects, each with ALL fields listed in the schema (use exact key names from the schema)
   - `totals` object with the 4 total fields
   - `recommendation` object with `decision` (the exact slug) and `justification` (a brief string explaining the numbers)

   Use `json.dump` with `indent=2` for readability.

7. **Write `/root/refill_summary.md`** with 4-8 lines that includes:
   - Total 90-day revenue (USD)
   - Total 100-day revenue (USD)
   - Absolute difference (USD)
   - Final decision using the exact slug (`switch_to_100_day` or `keep_90_day`)

### Step 3: Run the Script
```
python3 /root/analyze.py
```

### Step 4: Validate Outputs

1. Read and display `/root/refill_analysis.json` — verify:
   - It has exactly the keys from the schema
   - `medications` array has 10 entries
   - All currency values are rounded to 2 decimals
   - The `decision` field is one of the two exact slugs
   - The `assumptions` block matches the fixed parameters exactly

2. Read and display `/root/refill_summary.md` — verify:
   - 4-8 lines
   - Contains total 90-day revenue, total 100-day revenue, absolute difference, and the decision slug

3. Verify the JSON is valid by running:
```
python3 -c "import json; d=json.load(open('/root/refill_analysis.json')); print('Valid JSON, medications count:', len(d['medications'])); print('Decision:', d['recommendation']['decision'])"
```

### Important Notes
- The medication name field in the JSON must match exactly what appears in the CSV files. Do NOT rename, normalize case, or alter medication names.
- The `vial_size_drams` field should come from the vial_price.csv if available; include it as an integer.
- The `reimbursement_per_fill_300_patients_usd` is the value directly from reimbursement.csv (it already represents reimbursement for 300 patients per fill).
- Pay careful attention to the merge keys — the medication name columns may differ slightly across CSVs. Inspect and handle accordingly.
- Do NOT invent data. All 10 medications and their values must come from the CSV files.

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