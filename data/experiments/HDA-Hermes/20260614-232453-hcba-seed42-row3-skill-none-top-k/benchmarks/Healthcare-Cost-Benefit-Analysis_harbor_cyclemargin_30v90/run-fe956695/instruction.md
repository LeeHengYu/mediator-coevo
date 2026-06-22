# Task Instruction

## Task: Healthcare Cost-Benefit Analysis — 30-day vs 90-day Refill Cycle Margin Comparison

### Step 1: Inspect Input Files
Read and display the contents of all three input CSV files:
```
cat /root/acquisition_cost.csv
cat /root/packaging_cost.csv
cat /root/reimbursement.csv
```
Understand the columns, therapy names, and how they join together (likely by therapy name and/or canister_size_units).

### Step 2: Write a Python Script to Perform the Analysis
Create `/root/solve.py` that does the following:

1. **Load CSVs** using the `csv` module (or `pandas` if available):
   - `acquisition_cost.csv` — contains at minimum: `therapy`, `price_per_1000_doses_usd`
   - `packaging_cost.csv` — contains at minimum: `therapy`, `canister_size_units`, `packaging_cost_usd`
   - `reimbursement.csv` — contains at minimum: `therapy`, `reimbursement_per_fill_240_patients_usd`

2. **Merge/join** all three on `therapy`. For packaging cost, match by `canister_size_units` as well if needed (inspect the data first — there may be a direct therapy-level match).

3. **Constants:**
   - `patients_per_therapy = 240`
   - `doses_per_fill_30 = 60`, `fills_per_year_30 = 12`
   - `doses_per_fill_90 = 180`, `fills_per_year_90 = 4`
   - `switch_threshold_usd = 12000`

4. **Per-therapy calculations (for both 30-day and 90-day models):**
   - `annual_drug_cost = (price_per_1000_doses_usd / 1000) * doses_per_fill * fills_per_year * patients_per_therapy`
   - `annual_packaging_cost = packaging_cost_usd * fills_per_year * patients_per_therapy`
   - `annual_reimbursement = reimbursement_per_fill_240_patients_usd * fills_per_year`
     (Note: reimbursement is already for 240 patients per fill, so just multiply by fills_per_year)
   - `annual_margin = annual_reimbursement - annual_drug_cost - annual_packaging_cost`
   - `annual_margin_difference_90_minus_30 = annual_margin_90 - annual_margin_30`

5. **Totals:**
   - Sum all per-therapy `annual_margin_30_day_usd` → `total_annual_margin_30_day_usd`
   - Sum all per-therapy `annual_margin_90_day_usd` → `total_annual_margin_90_day_usd`
   - `total_annual_margin_difference_90_minus_30_usd` = total_90 - total_30
   - `absolute_total_margin_difference_usd` = abs(total_difference)

6. **Decision rule:**
   - If `absolute_total_margin_difference_usd < 12000` → `adopt_90_day`
   - Otherwise → `keep_30_day`

7. **Round all currency values to 2 decimal places.**

8. **Sort therapies alphabetically by `therapy` name.**

9. **Output `/root/cycle_margin_analysis.json`** matching the exact schema provided. Use `json.dump` with `indent=2`.

10. **Output `/root/cycle_margin_summary.md`** with 4–8 non-empty lines including:
    - Total 30-day margin (USD)
    - Total 90-day margin (USD)
    - Absolute difference (USD)
    - Final decision using exact slug `adopt_90_day` or `keep_30_day`

### Step 3: Run the Script
```
python3 /root/solve.py
```

### Step 4: Validate Outputs
```
cat /root/cycle_margin_analysis.json
cat /root/cycle_margin_summary.md
```

Verify:
- JSON is valid and parseable.
- `assumptions` block matches the constants exactly.
- `therapies` array is sorted alphabetically by `therapy`.
- All currency fields are rounded to 2 decimals.
- `totals` fields are correct sums.
- `absolute_total_margin_difference_usd` equals `abs(total_annual_margin_difference_90_minus_30_usd)`.
- Decision follows the rule: `< 12000` → `adopt_90_day`, otherwise `keep_30_day`.
- The summary `.md` has 4–8 non-empty lines and contains the required info with the exact decision slug.
- Verify a sample therapy's math manually: e.g., pick one therapy, compute drug cost by hand, and confirm it matches.

### Important Notes
- The drug cost formula uses `price_per_1000_doses_usd` — divide by 1000 to get price per dose.
- Packaging cost is per patient per fill.
- Reimbursement is per fill for all 240 patients (not per patient).
- Do NOT change the threshold or decision logic.
- Ensure the `justification` string in the recommendation is a brief, clear sentence explaining the decision.

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
Task metadata: author_email=gpt54@example.com, author_name=GPT-5.4, category=financial-analysis, difficulty=medium, tags=[healthcare, unit-economics, csv, json, decision-analysis].
Verifier config: timeout_sec=900.0.