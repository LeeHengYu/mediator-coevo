# Task Instruction

## Task: Healthcare Syncpack 28-day vs 56-day Cost-Benefit Analysis

### Step 1: Inspect Input Files
Read and display the contents of all three input CSV files:
```
cat /root/ingredient_cost.csv
cat /root/card_cost.csv
cat /root/reimbursement.csv
```
Understand the columns, medication names, and how they join together.

### Step 2: Write a Python script to produce the outputs
Create `/root/solve.py` that does the following:

1. **Read CSVs** using the `csv` module:
   - `ingredient_cost.csv` — columns include `medication` and `price_per_1000_capsules_usd`
   - `card_cost.csv` — columns include `blister_card_count` and `card_cost_usd`
   - `reimbursement.csv` — columns include `medication` and a reimbursement-per-cycle column for 180 patients

2. **Constants** (use these EXACT values):
   - `patients_per_medication = 180`
   - `fills_per_year_28_day = 12`
   - `fills_per_year_56_day = 6`
   - `capsules_per_fill_28_day = 56`
   - `capsules_per_fill_56_day = 112`
   - `switch_threshold_usd = 9000`
   - Dosing is 2 capsules/day but do NOT include a 'dosing' key in the assumptions block.

3. **Per-medication calculations** (join ingredient_cost and reimbursement on medication; join card_cost on blister_card_count):
   - `annual_drug_cost_28_day_usd = (price_per_1000_capsules_usd / 1000) * capsules_per_fill_28_day * fills_per_year_28_day * patients_per_medication`
   - `annual_drug_cost_56_day_usd = (price_per_1000_capsules_usd / 1000) * capsules_per_fill_56_day * fills_per_year_56_day * patients_per_medication`
   - Note: Both 28-day and 56-day drug costs should be identical (56*12 = 112*6 = 672 capsules/patient/year * 180 patients).
   - `annual_packaging_cost_28_day_usd = card_cost_usd * patients_per_medication * fills_per_year_28_day`
   - `annual_packaging_cost_56_day_usd = card_cost_usd * patients_per_medication * fills_per_year_56_day`
   - `annual_reimbursement_28_day_usd = reimbursement_per_cycle_180_patients * fills_per_year_28_day`
   - `annual_reimbursement_56_day_usd = reimbursement_per_cycle_180_patients * fills_per_year_56_day`
   - `annual_margin_28_day_usd = annual_reimbursement_28_day_usd - annual_drug_cost_28_day_usd - annual_packaging_cost_28_day_usd`
   - `annual_margin_56_day_usd = annual_reimbursement_56_day_usd - annual_drug_cost_56_day_usd - annual_packaging_cost_56_day_usd`
   - `annual_margin_difference_56_minus_28_usd = annual_margin_56_day_usd - annual_margin_28_day_usd`

4. **Round ALL currency values to 2 decimal places** using `round(value, 2)`.

5. **Sort medications alphabetically** by the `medication` field (case-sensitive sort; use Python default string sort).

6. **Totals**:
   - `total_annual_margin_28_day_usd` = sum of all per-med 28-day margins
   - `total_annual_margin_56_day_usd` = sum of all per-med 56-day margins
   - `total_annual_margin_difference_56_minus_28_usd` = sum of all per-med differences
   - `absolute_total_margin_difference_usd` = abs(total_annual_margin_difference_56_minus_28_usd)
   - Round all totals to 2 decimals.

7. **Decision rule**:
   - If `absolute_total_margin_difference_usd < 9000`: decision = `"convert_to_56_day"`
   - Otherwise: decision = `"keep_28_day"`

8. **Build the JSON object** with EXACTLY these top-level keys: `assumptions`, `medications`, `totals`, `recommendation`.
   - `assumptions` must contain EXACTLY these 6 keys (no more, no less): `patients_per_medication`, `fills_per_year_28_day`, `fills_per_year_56_day`, `capsules_per_fill_28_day`, `capsules_per_fill_56_day`, `switch_threshold_usd`.
   - `recommendation` must be a DICTIONARY (not a string) with keys `decision` (string slug) and `justification` (a brief human-readable string).
   - Each medication object must have EXACTLY these 14 keys: `medication`, `price_per_1000_capsules_usd`, `blister_card_count`, `card_cost_usd`, `reimbursement_per_cycle_180_patients_usd`, `annual_drug_cost_28_day_usd`, `annual_drug_cost_56_day_usd`, `annual_packaging_cost_28_day_usd`, `annual_packaging_cost_56_day_usd`, `annual_reimbursement_28_day_usd`, `annual_reimbursement_56_day_usd`, `annual_margin_28_day_usd`, `annual_margin_56_day_usd`, `annual_margin_difference_56_minus_28_usd`.

9. **Write `/root/syncpack_analysis.json`** with `json.dump(..., indent=2)`.

10. **Write `/root/syncpack_summary.md`** with 4-8 non-empty lines including:
    - Total 28-day margin in USD
    - Total 56-day margin in USD
    - Absolute difference in USD
    - The exact decision slug (`convert_to_56_day` or `keep_28_day`)

### Step 3: Run the script
```
python3 /root/solve.py
```

### Step 4: Validate outputs
1. Read and display `/root/syncpack_analysis.json` — verify:
   - `assumptions` has exactly 6 keys, no `dosing` key
   - `capsules_per_fill_28_day` is 56 (not 28)
   - `fills_per_year_28_day` is 12
   - `recommendation` is a dict with `decision` and `justification` keys
   - `medications` is sorted alphabetically
   - All currency values are rounded to 2 decimals
2. Read and display `/root/syncpack_summary.md` — verify 4-8 non-empty lines with the required info.
3. Verify JSON is valid: `python3 -c "import json; json.load(open('/root/syncpack_analysis.json'))"`

### Critical Reminders from Previous Failure
- Do NOT include a `dosing` key in the `assumptions` block.
- `capsules_per_fill_28_day` MUST be 56 (not 28).
- `fills_per_year_28_day` MUST be 12.
- `recommendation` MUST be a dictionary `{"decision": "...", "justification": "..."}`, NOT a plain string.
- Use the EXACT field names specified in the schema. Do not abbreviate or rename any keys.

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