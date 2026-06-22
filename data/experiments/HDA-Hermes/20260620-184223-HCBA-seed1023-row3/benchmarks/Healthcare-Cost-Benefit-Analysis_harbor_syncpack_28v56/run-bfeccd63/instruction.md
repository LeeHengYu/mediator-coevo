# Task Instruction

Execute the following steps in order:

1. **Inspect the input files** to understand their structure and contents:
 ```
 cat /root/ingredient_cost.csv
 cat /root/card_cost.csv
 cat /root/reimbursement.csv
 ```

2. **Write and run a Python script** (`/root/solve.py`) that performs the full analysis. The script must:

 a. Read all three CSV files using the `csv` module (or `pandas`).

 b. For each medication, join the data across the three files by medication name. Extract:
 - `price_per_1000_capsules_usd` from `ingredient_cost.csv`
 - `blister_card_count` and `card_cost_usd` from `card_cost.csv` (match by `blister_card_count` — note: the card_cost.csv likely has card costs keyed by blister_card_count; each medication's blister_card_count comes from one of the files, possibly ingredient_cost or a column in card_cost; inspect carefully)
 - `reimbursement_per_cycle_180_patients_usd` from `reimbursement.csv`

 c. Use these constants:
 - `patients_per_medication = 180`
 - `capsules_per_fill_28_day = 56`, `fills_per_year_28_day = 12`
 - `capsules_per_fill_56_day = 112`, `fills_per_year_56_day = 6`
 - `dosing = 2 capsules daily`
 - `switch_threshold_usd = 9000`

 d. For each medication compute:
 - `annual_drug_cost_28_day = (price_per_1000_capsules / 1000) * capsules_per_fill_28_day * fills_per_year_28_day * patients_per_medication`
 - `annual_drug_cost_56_day = (price_per_1000_capsules / 1000) * capsules_per_fill_56_day * fills_per_year_56_day * patients_per_medication`
   (Note: both should give the same total capsules/year = 56*12 = 112*6 = 672 per patient, so drug costs should be equal.)
 - `annual_packaging_cost_28_day = card_cost_usd * patients_per_medication * fills_per_year_28_day`
 - `annual_packaging_cost_56_day = card_cost_usd * patients_per_medication * fills_per_year_56_day`
   (IMPORTANT: The card_cost_usd is per patient per fill. For the 56-day model, you need to look up the card cost for the appropriate blister_card_count. The 28-day blister card count might differ from the 56-day one. Inspect the CSV carefully: if `card_cost.csv` has rows for different blister_card_counts, then for 28-day use the card count matching 56 capsules and for 56-day use the card count matching 112 capsules. Or if each medication row in the data specifies its own blister_card_count, use that. Read the files first to determine the correct join logic.)
 - `annual_reimbursement_28_day = reimbursement_per_cycle * fills_per_year_28_day`
 - `annual_reimbursement_56_day = reimbursement_per_cycle * fills_per_year_56_day`
 - `annual_margin = annual_reimbursement - annual_drug_cost - annual_packaging_cost` (for each model)
 - `annual_margin_difference = margin_56 - margin_28`

 e. Round ALL currency values to 2 decimal places.

 f. Sort the medications array alphabetically by medication name (case-insensitive sort is fine, but standard alphabetical).

 g. Compute totals:
 - `total_annual_margin_28_day_usd` = sum of all medications' 28-day margins
 - `total_annual_margin_56_day_usd` = sum of all medications' 56-day margins
 - `total_annual_margin_difference_56_minus_28_usd` = sum of all per-medication differences
 - `absolute_total_margin_difference_usd` = abs(total_difference)

 h. Apply decision rule:
 - If `abs(total_difference) < 9000` → `convert_to_56_day`
 - Otherwise → `keep_28_day`

 i. Write `/root/syncpack_analysis.json` with the exact schema specified (use `json.dump` with `indent=2`).

 j. Write `/root/syncpack_summary.md` with 4–8 non-empty lines including:
 - Total 28-day margin (USD)
 - Total 56-day margin (USD)
 - Absolute difference (USD)
 - The exact decision slug (`convert_to_56_day` or `keep_28_day`)

3. **Run the script**:
 ```
 python3 /root/solve.py
 ```

4. **Validate the outputs**:
 ```
 cat /root/syncpack_analysis.json
 cat /root/syncpack_summary.md
 python3 -c "import json; d=json.load(open('/root/syncpack_analysis.json')); print('medications count:', len(d['medications'])); print('sorted check:', [m['medication'] for m in d['medications']]); print('totals:', d['totals']); print('recommendation:', d['recommendation'])"
 ```

 Verify:
 - JSON is valid and parseable
 - `medications` array is sorted alphabetically by `medication`
 - All currency values have at most 2 decimal places
 - The `assumptions` block matches the required constants exactly
 - The summary markdown has 4–8 non-empty lines and contains the required information with the exact decision slug
 - The decision logic is correct: `abs(total_difference) < 9000` → `convert_to_56_day`, else `keep_28_day`

**CRITICAL NOTES on packaging cost logic**: After inspecting the CSV files, pay special attention to how `card_cost.csv` is structured. If it maps `blister_card_count` to `card_cost_usd`, then:
- For the 28-day model, each fill uses some number of blister cards (e.g., if blister_card_count=28, then 56 capsules needs 2 cards per fill; if blister_card_count=56, needs 1 card). However, re-read the task: it says packaging cost uses `card_cost_usd` from `card_cost.csv` **per patient per fill, matched by blister_card_count**. This means each medication has a `blister_card_count` attribute, and you look up the `card_cost_usd` for that count. The cost is per patient per fill — one flat cost per fill regardless of model. So both 28-day and 56-day use the SAME card_cost_usd per fill, but differ in number of fills (12 vs 6).
- The output schema has a single `blister_card_count` and `card_cost_usd` per medication, confirming this interpretation.

Inspect the files first, then code accordingly.

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