# Task Instruction

## Task: Healthcare Syncpack 28-day vs 56-day Cost-Benefit Analysis

### Step 1: Inspect input files
Read and display the contents of:
- `/root/ingredient_cost.csv`
- `/root/card_cost.csv`
- `/root/reimbursement.csv`

### Step 2: Compute the analysis using Python
Write and run a Python script that:

1. **Reads the three CSV files** using the `csv` module.
2. **Joins data by medication name** across the three files.
3. **For each medication**, computes (all values rounded to 2 decimals):
   - `annual_drug_cost_28_day_usd` = (price_per_1000_capsules_usd / 1000) × 56 capsules × 12 fills × 180 patients
   - `annual_drug_cost_56_day_usd` = (price_per_1000_capsules_usd / 1000) × 112 capsules × 6 fills × 180 patients
   - `annual_packaging_cost_28_day_usd` = card_cost_usd × 180 patients × 12 fills
   - `annual_packaging_cost_56_day_usd` = card_cost_usd × 180 patients × 6 fills
   - `annual_reimbursement_28_day_usd` = reimbursement_per_cycle_180_patients_usd × 12
   - `annual_reimbursement_56_day_usd` = reimbursement_per_cycle_180_patients_usd × 6
   - `annual_margin_28_day_usd` = annual_reimbursement_28_day - annual_drug_cost_28_day - annual_packaging_cost_28_day
   - `annual_margin_56_day_usd` = annual_reimbursement_56_day - annual_drug_cost_56_day - annual_packaging_cost_56_day
   - `annual_margin_difference_56_minus_28_usd` = annual_margin_56_day - annual_margin_28_day

   **Note on drug cost**: Both 28-day and 56-day models use the same total capsules per year (56×12 = 112×6 = 672 per patient), so annual drug costs should be equal. Compute them independently per the formulas anyway.

4. **Sort medications alphabetically** by medication name (case-insensitive sort, but preserve original casing).

5. **Compute totals**:
   - `total_annual_margin_28_day_usd` = sum of all annual_margin_28_day_usd
   - `total_annual_margin_56_day_usd` = sum of all annual_margin_56_day_usd
   - `total_annual_margin_difference_56_minus_28_usd` = sum of all per-medication differences
   - `absolute_total_margin_difference_usd` = abs(total_annual_margin_difference_56_minus_28_usd)

6. **Decision rule**:
   - If `absolute_total_margin_difference_usd < 9000`, decision = `"convert_to_56_day"`
   - Otherwise, decision = `"keep_28_day"`

7. **Build the JSON output** with EXACTLY this top-level structure:
   ```json
   {
     "assumptions": { ... },
     "medications": [ ... ],
     "totals": { ... },
     "recommendation": {
       "decision": "convert_to_56_day" or "keep_28_day",
       "justification": "<a brief string explaining the decision>"
     }
   }
   ```
   **CRITICAL**: The `decision` and `justification` keys MUST be nested inside a `recommendation` object. Do NOT put them at the top level. This was the exact failure in the previous run.

8. **Write `/root/syncpack_analysis.json`** with `json.dump(..., indent=2)`. All currency floats rounded to 2 decimals.

9. **Write `/root/syncpack_summary.md`** with 4-8 non-empty lines containing:
   - Total 28-day margin (USD)
   - Total 56-day margin (USD)
   - Absolute difference (USD)
   - The exact decision slug (`convert_to_56_day` or `keep_28_day`)

### Step 3: Validate
- Re-read `/root/syncpack_analysis.json` and confirm:
  - Top-level keys are exactly: `assumptions`, `medications`, `totals`, `recommendation`
  - `recommendation` is a dict with keys `decision` and `justification`
  - `medications` is sorted alphabetically
  - All currency values are rounded to 2 decimals
- Re-read `/root/syncpack_summary.md` and confirm it has 4-8 non-empty lines with the required content.

### Key Reminders
- Match `card_cost.csv` entries by `blister_card_count` column to the appropriate card count for each medication.
- The `reimbursement.csv` gives reimbursement per cycle for 180 patients — multiply by fills/year for annual.
- Round each individual currency value to 2 decimal places.
- The JSON schema must be followed exactly — no extra top-level keys, no missing nested keys.

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