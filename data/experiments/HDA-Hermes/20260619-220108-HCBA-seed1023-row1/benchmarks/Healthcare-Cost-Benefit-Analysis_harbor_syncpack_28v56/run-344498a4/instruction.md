# Task Instruction

## Task: Healthcare Cost-Benefit Analysis — 28-day vs 56-day Syncpack

### Step 1: Inspect Input Files
Read and display the contents of:
- `/root/ingredient_cost.csv`
- `/root/card_cost.csv`
- `/root/reimbursement.csv`

Also read `/tests/test_outputs.py` to understand the exact verifier expectations.

### Step 2: Create `/root/solve.py`
Write a Python script that:

1. **Reads the three CSV files** using the `csv` module.
2. **Joins data** by medication name across the three files.
3. **Uses these exact constants:**
   - `patients_per_medication = 180`
   - `capsules_per_day = 2`
   - `capsules_per_fill_28_day = 56`
   - `capsules_per_fill_56_day = 112`
   - `fills_per_year_28_day = 12`
   - `fills_per_year_56_day = 6`
   - `switch_threshold_usd = 9000`

4. **For each medication, computes:**
   - `annual_drug_cost_28_day_usd = (price_per_1000_capsules_usd / 1000) * capsules_per_fill_28_day * fills_per_year_28_day * patients_per_medication`
   - `annual_drug_cost_56_day_usd = (price_per_1000_capsules_usd / 1000) * capsules_per_fill_56_day * fills_per_year_56_day * patients_per_medication`
   - Note: both drug costs should be identical (56*12 == 112*6 == 672 capsules/patient/year).
   - `annual_packaging_cost_28_day_usd = card_cost_usd * patients_per_medication * fills_per_year_28_day`
   - `annual_packaging_cost_56_day_usd = card_cost_usd * patients_per_medication * fills_per_year_56_day`
     - **Card cost matching**: look up `card_cost_usd` from `card_cost.csv` by matching `blister_card_count` column. For 28-day fills the blister card count is 56; for 56-day fills it is 112. If different card sizes have different costs, use the appropriate cost for each model. Re-read the CSV carefully to determine if there's one card cost per medication or per blister size.
   - `annual_reimbursement_28_day_usd = reimbursement_per_cycle_180_patients_usd * fills_per_year_28_day`
   - `annual_reimbursement_56_day_usd = reimbursement_per_cycle_180_patients_usd * fills_per_year_56_day`
   - `annual_margin_28_day_usd = annual_reimbursement_28_day_usd - annual_drug_cost_28_day_usd - annual_packaging_cost_28_day_usd`
   - `annual_margin_56_day_usd = annual_reimbursement_56_day_usd - annual_drug_cost_56_day_usd - annual_packaging_cost_56_day_usd`
   - `annual_margin_difference_56_minus_28_usd = annual_margin_56_day_usd - annual_margin_28_day_usd`
   - All currency values rounded to 2 decimal places.

5. **Sorts medications alphabetically** by `medication` name.

6. **Computes totals:**
   - `total_annual_margin_28_day_usd` = sum of all `annual_margin_28_day_usd`
   - `total_annual_margin_56_day_usd` = sum of all `annual_margin_56_day_usd`
   - `total_annual_margin_difference_56_minus_28_usd` = sum of all per-med differences
   - `absolute_total_margin_difference_usd` = abs(total_annual_margin_difference_56_minus_28_usd)

7. **Decision rule:**
   - If `absolute_total_margin_difference_usd < 9000` → `convert_to_56_day`
   - Otherwise → `keep_28_day`

8. **Writes `/root/syncpack_analysis.json`** with **exactly** these keys (no abbreviations, no extras):
   ```
   assumptions.patients_per_medication
   assumptions.fills_per_year_28_day
   assumptions.fills_per_year_56_day
   assumptions.capsules_per_fill_28_day
   assumptions.capsules_per_fill_56_day
   assumptions.switch_threshold_usd
   medications[].medication
   medications[].price_per_1000_capsules_usd
   medications[].blister_card_count
   medications[].card_cost_usd
   medications[].reimbursement_per_cycle_180_patients_usd
   medications[].annual_drug_cost_28_day_usd
   medications[].annual_drug_cost_56_day_usd
   medications[].annual_packaging_cost_28_day_usd
   medications[].annual_packaging_cost_56_day_usd
   medications[].annual_reimbursement_28_day_usd
   medications[].annual_reimbursement_56_day_usd
   medications[].annual_margin_28_day_usd
   medications[].annual_margin_56_day_usd
   medications[].annual_margin_difference_56_minus_28_usd
   totals.total_annual_margin_28_day_usd
   totals.total_annual_margin_56_day_usd
   totals.total_annual_margin_difference_56_minus_28_usd
   totals.absolute_total_margin_difference_usd
   recommendation.decision
   recommendation.justification
   ```
   **CRITICAL**: Use the full key names above verbatim. Do NOT abbreviate (e.g., do NOT use `patients` instead of `patients_per_medication`, do NOT use `fills_28` instead of `fills_per_year_28_day`, etc.).

9. **Writes `/root/syncpack_summary.md`** with 4–8 non-empty lines including:
   - Total 28-day margin (USD)
   - Total 56-day margin (USD)
   - Absolute difference (USD)
   - Final decision using the exact slug `convert_to_56_day` or `keep_28_day`

### Step 3: Run the script
```bash
cd /root && python solve.py
```

### Step 4: Validate outputs
1. `cat /root/syncpack_analysis.json` — verify schema key names match exactly.
2. `cat /root/syncpack_summary.md` — verify 4-8 non-empty lines with required content.
3. Run the test suite:
```bash
cd / && python -m pytest tests/test_outputs.py -v
```

### Step 5: Fix any failures
If any test fails, read the error message carefully, re-read the relevant file, fix the issue in `solve.py`, re-run, and re-test. Repeat until all tests pass.

**Important notes from prior failures to avoid:**
- The `blister_card_count` field in each medication entry: check the CSV to determine what value to use. The task says packaging cost is matched by `blister_card_count` — inspect `card_cost.csv` to understand the structure.
- The `card_cost_usd` in the medication entry should reflect the card cost used. Check if 28-day and 56-day use different card costs (different blister counts) or the same.
- Do NOT invent key names. Copy them exactly from the schema above.
- The `assumptions` values must be the integer/number constants, not computed values.

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