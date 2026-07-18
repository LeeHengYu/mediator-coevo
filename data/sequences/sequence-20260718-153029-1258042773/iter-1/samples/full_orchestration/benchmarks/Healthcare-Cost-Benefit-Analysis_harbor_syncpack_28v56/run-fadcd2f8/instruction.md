# Task Instruction

Execute the following steps in order:

1. **Inspect the input files** to understand their structure and contents:
 ```
 cat /root/ingredient_cost.csv
 cat /root/card_cost.csv
 cat /root/reimbursement.csv
 ```

2. **Write and run a Python script** (`/root/solve.py`) that:

 a. Reads the three CSV files using the `csv` module.

 b. Joins/merges the data by medication name. Each medication should appear in all three files.

 c. For each medication, computes (all values are floats, rounded to 2 decimals at the end):

 - `price_per_1000_capsules_usd` from `ingredient_cost.csv`
 - `blister_card_count` and `card_cost_usd` from `card_cost.csv`
 - `reimbursement_per_cycle_180_patients_usd` from `reimbursement.csv`

 - **Drug cost per year:**
 - 28-day: `(price_per_1000_capsules_usd / 1000) * 56 * 180 * 12`
 - 56-day: `(price_per_1000_capsules_usd / 1000) * 112 * 180 * 6`
 - (These should be identical since 56*12 == 112*6 == 672 capsules/patient/year × 180 patients.)

 - **Packaging cost per year:**
 - 28-day: `card_cost_usd * 180 * 12`
 - 56-day: `card_cost_usd * 180 * 6`
 - Note: packaging cost is per patient per fill. The `card_cost_usd` is looked up by matching `blister_card_count`. For the 28-day model the blister card count is 56 capsules; for the 56-day model it is 112 capsules. **Important**: look at the `card_cost.csv` file carefully. If it has rows for different `blister_card_count` values (e.g., 56 and 112), then for each medication, look up the card cost for blister_card_count=56 for the 28-day model and blister_card_count=112 for the 56-day model. If the CSV has a single card cost per medication with a single blister_card_count, use that for both models. Inspect the file first to determine which case applies.

 - **Annual reimbursement:**
 - 28-day: `reimbursement_per_cycle * 12`
 - 56-day: `reimbursement_per_cycle * 6`
 - Again, check if reimbursement.csv has separate rows per cycle length. If it does, use the appropriate one. If it has one value per medication, use that value for both.

 - **Annual margin** = `annual_reimbursement - annual_drug_cost - annual_packaging_cost` (for each model)

 - **Margin difference** = `annual_margin_56_day - annual_margin_28_day`

 d. Sorts medications alphabetically by medication name.

 e. Computes totals:
 - `total_annual_margin_28_day_usd` = sum of all 28-day margins
 - `total_annual_margin_56_day_usd` = sum of all 56-day margins
 - `total_annual_margin_difference_56_minus_28_usd` = sum of all per-medication differences
 - `absolute_total_margin_difference_usd` = abs(total_difference)

 f. Decision:
 - If `absolute_total_margin_difference_usd < 9000` → `"convert_to_56_day"`
 - Otherwise → `"keep_28_day"`

 g. Rounds ALL currency values to 2 decimal places.

 h. Writes `/root/syncpack_analysis.json` with the exact schema shown in the task (including nested `"recommendation"` object with `"decision"` and `"justification"` keys). Use `json.dump` with `indent=2`.

 i. Writes `/root/syncpack_summary.md` with 4–8 non-empty lines containing:
 - Total 28-day margin (USD)
 - Total 56-day margin (USD)
 - Absolute difference (USD)
 - Final decision using the exact slug `convert_to_56_day` or `keep_28_day`

3. **Run the script:**
 ```
 python3 /root/solve.py
 ```

4. **Validate the outputs:**
 ```
 cat /root/syncpack_analysis.json
 cat /root/syncpack_summary.md
 python3 -c "import json; d=json.load(open('/root/syncpack_analysis.json')); print('recommendation' in d); print(d['recommendation']); print(len(d['medications'])); print(d['totals'])"
 ```

 Confirm:
 - JSON parses without error
 - `recommendation` key exists and contains `decision` and `justification`
 - `medications` array is sorted alphabetically
 - All currency values have at most 2 decimal places
 - Summary markdown has 4–8 non-empty lines and includes the required values and slug

**Critical warnings from past failures:**
- The `recommendation` MUST be nested: `{"recommendation": {"decision": "...", "justification": "..."}}` — NOT at the root level.
- Pay close attention to the CSV structure before coding. The packaging cost may differ between 28-day and 56-day models if the CSV has rows for different blister card counts. Similarly for reimbursement.
- Drug cost for 28-day and 56-day should be identical (same total capsules/year), but packaging and reimbursement will differ, driving the margin difference.

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