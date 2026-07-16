# Task Instruction

Execute the following steps in order:

1. **Inspect the input files** to understand their structure:
 ```
 cat /root/ingredient_cost.csv
 cat /root/card_cost.csv
 cat /root/reimbursement.csv
 ```

2. **Write and run a Python script** (`/root/solve.py`) that:

 a. Reads all three CSVs using the `csv` module (or pandas if available).
 
 b. Joins/merges the data by medication name. The three files should share a medication identifier column.
 
 c. For each medication, computes (all values rounded to 2 decimals):
 - `annual_drug_cost_28_day_usd = (price_per_1000_capsules_usd / 1000) * 56 * 12 * 180`
 - `annual_drug_cost_56_day_usd = (price_per_1000_capsules_usd / 1000) * 112 * 6 * 180`
 - Note: both drug costs should be identical (56*12 == 112*6 == 672 capsules/patient/year * 180 patients).
 - `annual_packaging_cost_28_day_usd = card_cost_usd * 180 * 12` (one card per patient per fill)
 - `annual_packaging_cost_56_day_usd = card_cost_usd * 180 * 6` (one card per patient per fill, matched by blister_card_count — use the card cost row whose blister_card_count matches the medication's blister_card_count)
 - **IMPORTANT**: Carefully check how card_cost.csv is structured. It may have rows keyed by `blister_card_count` (e.g., 28 and 56). For the 28-day model, use the card cost for blister_card_count matching the medication's card count (likely 28). For the 56-day model, the blister_card_count may differ (likely 56). Read the CSV carefully to determine the correct matching. If the medication row itself has a `blister_card_count` field, that tells you which card size it currently uses. For the 56-day model, you'd use the card with double the blister count. **Re-read the task**: packaging cost uses `card_cost_usd` from `card_cost.csv` per patient per fill, matched by `blister_card_count`. So for 28-day fills you match on the 28-day blister card count, and for 56-day fills you match on the 56-day blister card count. Inspect the data to determine the exact mapping.
 - `annual_reimbursement_28_day_usd = reimbursement_per_cycle_180_patients_usd * 12`
 - `annual_reimbursement_56_day_usd = reimbursement_per_cycle_180_patients_usd * 6`
 - `annual_margin = annual_reimbursement - annual_drug_cost - annual_packaging_cost` (for each model)
 - `annual_margin_difference_56_minus_28_usd = annual_margin_56_day_usd - annual_margin_28_day_usd`
 
 d. Sorts medications alphabetically by medication name.
 
 e. Computes totals:
 - `total_annual_margin_28_day_usd` = sum of all 28-day margins
 - `total_annual_margin_56_day_usd` = sum of all 56-day margins
 - `total_annual_margin_difference_56_minus_28_usd` = sum of all per-medication differences
 - `absolute_total_margin_difference_usd` = abs(total_annual_margin_difference_56_minus_28_usd)
 
 f. Applies decision rule:
 - If `absolute_total_margin_difference_usd < 9000` → `convert_to_56_day`
 - Otherwise → `keep_28_day`
 
 g. Writes `/root/syncpack_analysis.json` with the exact schema from the task, all currency values rounded to 2 decimals.
 
 h. Writes `/root/syncpack_summary.md` with 4–8 non-empty lines including:
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
 python3 -c "import json; d=json.load(open('/root/syncpack_analysis.json')); print('Meds:', len(d['medications'])); print('Sorted:', [m['medication'] for m in d['medications']]); print('Totals:', d['totals']); print('Decision:', d['recommendation']['decision'])"
 cat /root/syncpack_summary.md
 ```
 Confirm:
 - JSON is valid and parseable
 - Medications are sorted alphabetically
 - All currency fields are rounded to 2 decimals
 - The summary has 4–8 non-empty lines with all required values
 - The decision slug matches exactly (`convert_to_56_day` or `keep_28_day`)

**Key caution**: The card_cost.csv likely has multiple rows for different blister card counts. You need to determine which card cost applies to the 28-day model vs the 56-day model. The task says packaging cost is matched by `blister_card_count`. Inspect the data carefully before coding the join logic. If each medication has its own `blister_card_count` in ingredient_cost.csv or reimbursement.csv, use that to look up the card cost. For the 56-day model, think about whether the blister card count doubles (e.g., 28→56) or stays the same. The task schema shows each medication has one `blister_card_count` and one `card_cost_usd`, suggesting a single card cost per medication used for both models. Read the data to confirm.

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