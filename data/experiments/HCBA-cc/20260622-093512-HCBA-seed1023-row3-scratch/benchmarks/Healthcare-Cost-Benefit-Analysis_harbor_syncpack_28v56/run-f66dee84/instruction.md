# Task Instruction

Execute the following steps in order:

1. **Inspect the input files** to understand their structure and contents:
 ```
 cat /root/ingredient_cost.csv
 cat /root/card_cost.csv
 cat /root/reimbursement.csv
 ```

2. **Write and run a Python script** (`/root/solve.py`) that performs the full analysis and produces both output files. The script must:

 a. **Read the three CSV files** using the `csv` module (or pandas if available).
 
 b. **Merge/join the data by medication name.** Each medication should appear in all three files. From each file extract:
 - `ingredient_cost.csv`: `medication`, `price_per_1000_capsules_usd`
 - `card_cost.csv`: `medication`, `blister_card_count`, `card_cost_usd`
 - `reimbursement.csv`: `medication`, `reimbursement_per_cycle_180_patients_usd`
 
 c. **For each medication, compute** (using the constants: patients=180, 28-day: 56 caps/fill, 12 fills/yr; 56-day: 112 caps/fill, 6 fills/yr):
 
 - `annual_drug_cost_28_day_usd = (price_per_1000_capsules_usd / 1000) * 56 * 180 * 12`
 - `annual_drug_cost_56_day_usd = (price_per_1000_capsules_usd / 1000) * 112 * 180 * 6`
 - Note: both drug costs should be identical (both equal `price_per_1000_capsules_usd / 1000 * 120960`). Compute them separately anyway.
 - `annual_packaging_cost_28_day_usd = card_cost_usd * 180 * 12`
 - `annual_packaging_cost_56_day_usd = card_cost_usd * 180 * 6`
 - **Important for packaging:** The `card_cost_usd` from `card_cost.csv` is the cost per patient per fill. The `blister_card_count` column is informational metadata to include in output but does NOT change the packaging cost formula.
 - `annual_reimbursement_28_day_usd = reimbursement_per_cycle_180_patients_usd * 12`
 - `annual_reimbursement_56_day_usd = reimbursement_per_cycle_180_patients_usd * 6`
 - `annual_margin_28_day_usd = annual_reimbursement_28_day_usd - annual_drug_cost_28_day_usd - annual_packaging_cost_28_day_usd`
 - `annual_margin_56_day_usd = annual_reimbursement_56_day_usd - annual_drug_cost_56_day_usd - annual_packaging_cost_56_day_usd`
 - `annual_margin_difference_56_minus_28_usd = annual_margin_56_day_usd - annual_margin_28_day_usd`
 - Round ALL currency values to 2 decimal places.
 
 d. **Sort medications alphabetically** by medication name (case-insensitive sort is safest, but match whatever the data contains).
 
 e. **Compute totals:**
 - `total_annual_margin_28_day_usd` = sum of all medications' `annual_margin_28_day_usd`
 - `total_annual_margin_56_day_usd` = sum of all medications' `annual_margin_56_day_usd`
 - `total_annual_margin_difference_56_minus_28_usd` = sum of all medications' `annual_margin_difference_56_minus_28_usd`
 - `absolute_total_margin_difference_usd` = `abs(total_annual_margin_difference_56_minus_28_usd)`
 - Round all totals to 2 decimal places.
 
 f. **Decision rule:**
 - If `absolute_total_margin_difference_usd < 9000`, decision = `"convert_to_56_day"`
 - Otherwise, decision = `"keep_28_day"`
 - Write a short justification string referencing the absolute difference and the $9,000 threshold.
 
 g. **Write `/root/syncpack_analysis.json`** with the exact schema from the task. Use `json.dumps` with `indent=2`. Ensure:
 - The `assumptions` block has the exact keys and values specified.
 - The `medications` array contains objects with ALL 14 specified fields.
 - The `totals` block has all 4 specified fields.
 - The `recommendation` block has `decision` and `justification`.
 
 h. **Write `/root/syncpack_summary.md`** with 4-8 non-empty lines containing:
 - Total 28-day margin in USD
 - Total 56-day margin in USD
 - Absolute difference in USD
 - The exact decision slug (`convert_to_56_day` or `keep_28_day`)
 - Example format:
 ```
 # Syncpack Analysis Summary
 
 Total annual margin (28-day cycle): $X.XX
 Total annual margin (56-day cycle): $Y.YY
 Absolute margin difference: $Z.ZZ
 Recommendation: convert_to_56_day
 ```

3. **Run the script:**
 ```
 python3 /root/solve.py
 ```

4. **Validate the outputs:**
 - `cat /root/syncpack_analysis.json` — confirm valid JSON, correct schema, medications sorted alphabetically, all values rounded to 2 decimals.
 - `cat /root/syncpack_summary.md` — confirm 4-8 non-empty lines with all required information and the exact decision slug.
 - Verify the JSON parses correctly: `python3 -c "import json; d=json.load(open('/root/syncpack_analysis.json')); print('meds:', len(d['medications'])); print('decision:', d['recommendation']['decision']); print('abs_diff:', d['totals']['absolute_total_margin_difference_usd'])"`

5. **If any validation fails**, inspect the error, fix the script, and re-run until both output files are correct.

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