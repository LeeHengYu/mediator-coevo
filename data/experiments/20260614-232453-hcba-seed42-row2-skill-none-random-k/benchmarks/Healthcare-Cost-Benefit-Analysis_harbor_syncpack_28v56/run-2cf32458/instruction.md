# Task Instruction

## Task: Healthcare Cost-Benefit Analysis – 28-day vs 56-day Syncpack

### Step 1: Inspect Input Files
Read and display the contents of all three input CSV files:
```
cat /root/ingredient_cost.csv
cat /root/card_cost.csv
cat /root/reimbursement.csv
```
Understand the columns, medications listed, and how they join together (likely by medication name or blister_card_count).

### Step 2: Write a Python Script to Perform the Analysis
Create `/root/solve.py` that does the following:

1. **Load CSVs** using the `csv` module (or `pandas` if available).
2. **Join data**: For each medication, pull:
   - `price_per_1000_capsules_usd` from `ingredient_cost.csv`
   - `card_cost_usd` from `card_cost.csv` matched by `blister_card_count` (the blister_card_count for each medication comes from `ingredient_cost.csv` or whichever file contains it — inspect to confirm)
   - `reimbursement_per_cycle_180_patients_usd` from `reimbursement.csv`
3. **Constants**:
   - patients = 180
   - capsules_per_day = 2
   - 28-day model: 56 capsules/fill, 12 fills/year
   - 56-day model: 112 capsules/fill, 6 fills/year
4. **Per-medication calculations** (all for 180 patients combined):
   - `annual_drug_cost_28_day = (price_per_1000_capsules / 1000) * 56 * 12 * 180`
   - `annual_drug_cost_56_day = (price_per_1000_capsules / 1000) * 112 * 6 * 180`
   - NOTE: Both drug costs should be identical (56*12 = 112*6 = 672 capsules/patient/year). Compute them separately anyway.
   - `annual_packaging_cost_28_day = card_cost_usd * 180 * 12`
   - `annual_packaging_cost_56_day = card_cost_usd * 180 * 6`
   - NOTE on packaging: The card_cost_usd is per patient per fill. For 56-day fills, look up the card cost matching the 56-day blister_card_count. **Important**: Check if `card_cost.csv` has entries for different blister_card_counts. The 28-day model uses a card with `blister_card_count` matching 56 capsules (or whatever the medication's card count is for 28-day), and the 56-day model may use a different card size. **Inspect the data carefully.** If there's only one card cost per medication regardless of cycle, use it for both. If `card_cost.csv` is keyed by `blister_card_count`, then:
     - For 28-day: look up card cost for the medication's blister_card_count
     - For 56-day: the blister_card_count might double, check if that entry exists
     - **If the CSV only has one blister_card_count per medication, use the same card_cost for both models.**
   - `annual_reimbursement_28_day = reimbursement_per_cycle * 12`
   - `annual_reimbursement_56_day = reimbursement_per_cycle * 6`
   - `annual_margin = annual_reimbursement - annual_drug_cost - annual_packaging_cost` (for each model)
   - `annual_margin_difference = margin_56 - margin_28`
5. **Round all currency values to 2 decimal places.**
6. **Sort medications alphabetically by medication name.**
7. **Totals**:
   - Sum all 28-day margins, all 56-day margins, compute total difference and absolute difference.
8. **Decision**:
   - If `abs(total_difference) < 9000`: recommend `convert_to_56_day`
   - Otherwise: recommend `keep_28_day`
   - Write a justification string like: `"The absolute margin difference of $X is below/above the $9,000 threshold, so we recommend [decision]."`
9. **Output `/root/syncpack_analysis.json`** with the exact schema specified. Use `json.dump` with `indent=2`.
10. **Output `/root/syncpack_summary.md`** with 4-8 non-empty lines including:
    - Total 28-day margin (USD)
    - Total 56-day margin (USD)
    - Absolute difference (USD)
    - Final decision using the exact slug `convert_to_56_day` or `keep_28_day`

### Step 3: Run the Script
```
python3 /root/solve.py
```

### Step 4: Validate Outputs
1. `cat /root/syncpack_analysis.json` — verify:
   - Valid JSON, parseable
   - `assumptions` block matches the constants exactly
   - `medications` array is sorted alphabetically
   - All currency fields are rounded to 2 decimals
   - `totals` block has all 4 fields
   - `recommendation` has `decision` and `justification`
2. `cat /root/syncpack_summary.md` — verify:
   - 4-8 non-empty lines
   - Contains total 28-day margin, total 56-day margin, absolute difference, and the exact decision slug
3. Spot-check one medication's math manually:
   - drug_cost = (price_per_1000 / 1000) * capsules_per_fill * fills * 180
   - packaging = card_cost * 180 * fills
   - reimbursement = reimb_per_cycle * fills
   - margin = reimbursement - drug_cost - packaging

### Important Notes
- The `blister_card_count` field appears in the output schema per medication. Inspect which input file provides this and how it links to `card_cost.csv`.
- Drug cost is for ALL 180 patients combined (not per patient).
- Packaging cost is per patient per fill, so multiply by 180 patients and fills/year.
- Reimbursement per cycle is already for 180 patients, so just multiply by fills/year.
- Do NOT invent data. Only use what's in the CSV files.
- If any ambiguity arises from the CSV structure, state what you found and make the most reasonable interpretation consistent with the schema.

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