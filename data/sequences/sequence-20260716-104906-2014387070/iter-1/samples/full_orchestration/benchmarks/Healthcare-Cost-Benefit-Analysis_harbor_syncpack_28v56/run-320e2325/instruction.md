# Task Instruction

## Task: Healthcare Cost-Benefit Analysis — 28-day vs 56-day Syncpack Comparison

### Objective
Read three CSV input files, compute annual margins for 28-day and 56-day medication synchronization card cycles, and produce two output files.

### Step-by-step Plan

#### 1. Inspect Input Files
- `cat /root/ingredient_cost.csv` — note columns: expect `medication`, `price_per_1000_capsules_usd`, and possibly `blister_card_count`.
- `cat /root/card_cost.csv` — note columns: expect `blister_card_count`, `card_cost_usd`.
- `cat /root/reimbursement.csv` — note columns: expect `medication`, `reimbursement_per_cycle_180_patients_usd` (or similar).

Record exact column names and all rows.

#### 2. Write a Python script `/root/solve.py` that does the following:

**a. Load CSVs** with `csv.DictReader`.

**b. Build lookup tables:**
- `card_cost_lookup`: `blister_card_count` → `card_cost_usd` (from card_cost.csv)
- `reimbursement_lookup`: `medication` → reimbursement per cycle for 180 patients (from reimbursement.csv)
- `ingredient_lookup`: `medication` → dict with `price_per_1000_capsules_usd` and `blister_card_count` (from ingredient_cost.csv)

**c. For each medication** (from ingredient_cost.csv), compute:

```
Constants:
  patients = 180
  fills_28 = 12,  fills_56 = 6
  caps_per_fill_28 = 56,  caps_per_fill_56 = 112

price_per_cap = price_per_1000_capsules_usd / 1000

annual_drug_cost_28 = price_per_cap * caps_per_fill_28 * fills_28 * patients
annual_drug_cost_56 = price_per_cap * caps_per_fill_56 * fills_56 * patients
  (Note: these should be equal — both = price_per_cap * 672 * 180 capsules/year)

card_cost = card_cost_lookup[blister_card_count]
annual_packaging_cost_28 = card_cost * patients * fills_28
annual_packaging_cost_56 = card_cost * patients * fills_56

reimbursement_per_cycle = reimbursement_lookup[medication]
annual_reimbursement_28 = reimbursement_per_cycle * fills_28
annual_reimbursement_56 = reimbursement_per_cycle * fills_56

annual_margin_28 = annual_reimbursement_28 - annual_drug_cost_28 - annual_packaging_cost_28
annual_margin_56 = annual_reimbursement_56 - annual_drug_cost_56 - annual_packaging_cost_56

margin_difference = annual_margin_56 - annual_margin_28
```

Round ALL currency values to 2 decimal places using `round(value, 2)`.

**d. Sort medications alphabetically** by `medication` name (case-sensitive sort; if unsure, use default Python string sort).

**e. Compute totals:**
```
total_margin_28 = sum of all annual_margin_28
total_margin_56 = sum of all annual_margin_56
total_difference = sum of all margin_difference  (equivalently: total_margin_56 - total_margin_28)
absolute_difference = abs(total_difference)
```
Round totals to 2 decimals.

**f. Decision rule:**
- If `absolute_difference < 9000`: decision = `"convert_to_56_day"`
- Otherwise: decision = `"keep_28_day"`

**g. Build the JSON structure** exactly matching the schema in the task (all field names must match exactly). Write to `/root/syncpack_analysis.json` with `json.dump(..., indent=2)`.

**h. Build `/root/syncpack_summary.md`** with 4–8 non-empty lines containing:
- Total 28-day margin (USD)
- Total 56-day margin (USD)
- Absolute difference (USD)
- The exact decision slug (`convert_to_56_day` or `keep_28_day`)

Example format:
```
# Syncpack Analysis Summary

Total annual margin (28-day): $X.XX
Total annual margin (56-day): $Y.YY
Absolute margin difference: $Z.ZZ
Recommendation: convert_to_56_day
```

#### 3. Run the script
```bash
python3 /root/solve.py
```

#### 4. Validate outputs
- `cat /root/syncpack_analysis.json` — verify JSON parses, all fields present, medications sorted alphabetically, all values are numbers rounded to 2 decimals.
- `cat /root/syncpack_summary.md` — verify 4–8 non-empty lines, contains all four required items with exact slug.
- Verify the JSON `assumptions` block matches the fixed constants exactly.
- Spot-check one medication's math manually.

### Key Pitfalls to Avoid
- Column name mismatches: use the EXACT column headers from the CSVs.
- Forgetting to convert `price_per_1000_capsules_usd` by dividing by 1000.
- Not rounding intermediate per-medication values to 2 decimals before summing (round each medication's values, then also round totals).
- Sorting: sort the medications list alphabetically by the `medication` field.
- The justification string in the recommendation should mention the threshold and the absolute difference.
- Ensure `blister_card_count` is used as the join key between ingredient_cost.csv and card_cost.csv (convert to int if needed for matching).
- The reimbursement CSV gives reimbursement per cycle for 180 patients — this is already for all 180 patients per cycle, so do NOT multiply by 180 again for reimbursement.

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