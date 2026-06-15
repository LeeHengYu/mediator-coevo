# Task Instruction

## Task: Retail Pharmacy Auto-Refill Cost-Benefit Analysis

You need to analyze whether a pharmacy should keep 90-day fills or switch to 100-day fills for its top 10 maintenance medications, then produce two output files.

### Step 1: Inspect Input Files

Read and display the contents of all three input CSV files:
```
cat /root/wholesale_price.csv
cat /root/vial_price.csv
cat /root/reimbursement.csv
```

Understand the column names and data types. Also optionally glance at the PDFs to cross-check if needed, but the CSVs are the authoritative machine-readable inputs.

### Step 2: Write a Python Script to Perform the Analysis

Create and run a Python script `/root/analyze.py` that does the following:

1. **Load all three CSVs** using pandas. Join/merge them on the medication name column (inspect the actual column names first — they may vary between files).

2. **For each of the 10 medications, compute:**

   - `annual_drug_cost_90_day = (price_per_1000_tablets / 1000) * 90 * 4 * 300`
   - `annual_drug_cost_100_day = (price_per_1000_tablets / 1000) * 100 * 3 * 300`
   - `annual_supply_cost_90_day = vial_price * 300 * 4`  (one vial per patient per fill)
   - `annual_supply_cost_100_day = vial_price * 300 * 3`
   - `annual_reimbursement_90_day = reimbursement_per_fill_300_patients * 4`
   - `annual_reimbursement_100_day = reimbursement_per_fill_300_patients * 3`
   - `annual_revenue_90_day = annual_reimbursement_90_day - annual_drug_cost_90_day - annual_supply_cost_90_day`
   - `annual_revenue_100_day = annual_reimbursement_100_day - annual_drug_cost_100_day - annual_supply_cost_100_day`
   - `annual_revenue_difference_100_minus_90 = annual_revenue_100_day - annual_revenue_90_day`

3. **Round all currency values to 2 decimal places.**

4. **Compute totals:**
   - `total_annual_revenue_90_day` = sum of all medications' `annual_revenue_90_day`
   - `total_annual_revenue_100_day` = sum of all medications' `annual_revenue_100_day`
   - `total_annual_revenue_difference_100_minus_90` = sum of all per-medication differences
   - `absolute_total_revenue_difference` = abs(total_annual_revenue_difference_100_minus_90)

5. **Decision rule:**
   - If `absolute_total_revenue_difference < 16000`, decision = `"switch_to_100_day"`
   - Otherwise, decision = `"keep_90_day"`

6. **Write `/root/refill_analysis.json`** with the exact schema specified below. The `medications` array must have one object per medication. All field names must match exactly:
   ```
   assumptions.patients_per_medication: 300
   assumptions.fills_per_year_90_day: 4
   assumptions.fills_per_year_100_day: 3
   assumptions.tablets_per_fill_90_day: 90
   assumptions.tablets_per_fill_100_day: 100
   assumptions.switch_threshold_usd: 16000
   ```
   Each medication object fields:
   - `medication` (string, the medication name)
   - `price_per_1000_tablets_usd`
   - `vial_size_drams` (integer)
   - `vial_price_usd`
   - `reimbursement_per_fill_300_patients_usd`
   - `annual_drug_cost_90_day_usd`
   - `annual_drug_cost_100_day_usd`
   - `annual_supply_cost_90_day_usd`
   - `annual_supply_cost_100_day_usd`
   - `annual_reimbursement_90_day_usd`
   - `annual_reimbursement_100_day_usd`
   - `annual_revenue_90_day_usd`
   - `annual_revenue_100_day_usd`
   - `annual_revenue_difference_100_minus_90_usd`

   Totals fields:
   - `total_annual_revenue_90_day_usd`
   - `total_annual_revenue_100_day_usd`
   - `total_annual_revenue_difference_100_minus_90_usd`
   - `absolute_total_revenue_difference_usd`

   Recommendation fields:
   - `decision`: exactly `"switch_to_100_day"` or `"keep_90_day"`
   - `justification`: a brief string explaining the decision referencing the threshold and the absolute difference

7. **Write `/root/refill_summary.md`** — a Markdown file, 4-8 lines, that includes:
   - Total 90-day revenue (USD)
   - Total 100-day revenue (USD)
   - Absolute difference (USD)
   - Final decision using the exact slug `switch_to_100_day` or `keep_90_day`

### Step 3: Run the Script

```bash
python3 /root/analyze.py
```

### Step 4: Validate Outputs

1. **Check JSON validity and schema:**
   ```bash
   python3 -c "import json; d=json.load(open('/root/refill_analysis.json')); print('medications count:', len(d['medications'])); print('totals:', d['totals']); print('decision:', d['recommendation']['decision'])"
   ```
   - Confirm there are exactly 10 medications.
   - Confirm all required keys exist.
   - Confirm decision is one of the two exact slugs.

2. **Check the summary:**
   ```bash
   cat /root/refill_summary.md
   ```
   - Confirm it's 4-8 lines.
   - Confirm it contains the total revenues, absolute difference, and exact decision slug.

3. **Spot-check one medication manually:**
   Pick the first medication from the JSON output. Manually verify its `annual_drug_cost_90_day_usd` = `(price_per_1000_tablets_usd / 1000) * 90 * 4 * 300` and similarly for other fields. Print the verification.

### Important Notes
- The medication name column may differ across CSVs — inspect actual column headers and normalize/match accordingly (e.g., strip whitespace, case-insensitive merge if needed).
- `vial_size_drams` should be an integer in the JSON output.
- `switch_threshold_usd` in assumptions should be the integer `16000`, not a float.
- All USD values in medications and totals must be floats rounded to 2 decimal places.
- Do NOT modify or weaken any part of the decision rule or formulas.

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