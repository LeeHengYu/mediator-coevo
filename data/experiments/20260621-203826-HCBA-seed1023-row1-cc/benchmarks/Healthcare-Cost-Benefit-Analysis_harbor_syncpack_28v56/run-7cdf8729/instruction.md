# Task Instruction

Execute the following steps in order:

1. **Inspect the input files** to understand their structure and contents:
 ```
 cat /root/ingredient_cost.csv
 cat /root/card_cost.csv
 cat /root/reimbursement.csv
 ```

2. **Write and run a Python script** (`/root/solve.py`) that performs the full analysis. The script must:

 a. **Read the three CSV files** using the `csv` module (or `pandas`):
 - `ingredient_cost.csv` — columns should include `medication` and `price_per_1000_capsules_usd`
 - `card_cost.csv` — columns should include `blister_card_count` and `card_cost_usd`
 - `reimbursement.csv` — columns should include `medication` and a reimbursement amount per cycle for 180 patients

 b. **For each medication**, compute:
 - **Annual drug cost** (same for both models since total capsules/year are the same):
 - 28-day: `(price_per_1000_capsules_usd / 1000) * 56 capsules/fill * 12 fills/year * 180 patients`
 - 56-day: `(price_per_1000_capsules_usd / 1000) * 112 capsules/fill * 6 fills/year * 180 patients`
 - Note: both should equal the same total (120,960 capsules/year × price), but compute them separately per the schema.
 - **Annual packaging cost**:
 - Look up `card_cost_usd` by matching the medication's `blister_card_count` in `card_cost.csv`.
 - 28-day: `card_cost_usd * 180 patients * 12 fills/year`
 - 56-day: `card_cost_usd * 180 patients * 6 fills/year`
 - IMPORTANT: The blister_card_count for 56-day will likely differ from 28-day. For 28-day cycles, each fill is 56 capsules — look up the card cost for blister_card_count matching 56 (or whatever the medication's card count is). For 56-day cycles, each fill is 112 capsules — look up the card cost for blister_card_count=112. **Re-read the CSV carefully**: if `card_cost.csv` has a `blister_card_count` column, the 28-day model uses the card matching 56 capsules and the 56-day model uses the card matching 112 capsules. However, if the CSV links medications to specific card counts, use that linkage. Inspect the CSV first to determine the correct join logic.
 - **Annual reimbursement**:
 - From `reimbursement.csv`, get `reimbursement_per_cycle_180_patients_usd` for each medication.
 - 28-day: `reimbursement_per_cycle * 12`
 - 56-day: `reimbursement_per_cycle * 6`
 - **Annual margin**: `annual_reimbursement - annual_drug_cost - annual_packaging_cost` (for each model)
 - **Margin difference**: `annual_margin_56_day - annual_margin_28_day`

 c. **Sort medications alphabetically** by `medication` name.

 d. **Compute totals**:
 - `total_annual_margin_28_day_usd` = sum of all medications' 28-day margins
 - `total_annual_margin_56_day_usd` = sum of all medications' 56-day margins
 - `total_annual_margin_difference_56_minus_28_usd` = sum of all per-medication differences
 - `absolute_total_margin_difference_usd` = `abs(total_annual_margin_difference_56_minus_28_usd)`

 e. **Decision rule**:
 - If `absolute_total_margin_difference_usd < 9000` → `convert_to_56_day`
 - Otherwise → `keep_28_day`

 f. **Round all currency values to 2 decimal places** in the output.

 g. **Write `/root/syncpack_analysis.json`** with the exact schema from the task, using `json.dump` with `indent=2`.

 h. **Write `/root/syncpack_summary.md`** with 4–8 non-empty lines containing:
 - Total 28-day margin (USD)
 - Total 56-day margin (USD)
 - Absolute difference (USD)
 - The exact decision slug: `convert_to_56_day` or `keep_28_day`

3. **Run the script**:
 ```
 python3 /root/solve.py
 ```

4. **Validate the outputs**:
 ```
 cat /root/syncpack_analysis.json
 cat /root/syncpack_summary.md
 python3 -c "import json; d=json.load(open('/root/syncpack_analysis.json')); print('Meds sorted:', [m['medication'] for m in d['medications']]); print('Decision:', d['recommendation']['decision']); print('Abs diff:', d['totals']['absolute_total_margin_difference_usd'])"
 ```
 - Confirm JSON is valid and parseable.
 - Confirm medications are sorted alphabetically.
 - Confirm the decision matches the threshold rule.
 - Confirm the summary markdown has 4–8 non-empty lines and includes all required values and the exact slug.

**CRITICAL NOTES**:
- Inspect the CSV files FIRST before writing the script. The join logic for card_cost depends on how the CSVs are structured. The medication may have a `blister_card_count` field in `ingredient_cost.csv` or `reimbursement.csv` that you use to look up the card cost.
- If `card_cost.csv` has multiple rows for different blister card counts (e.g., 56 and 112), you need to use the appropriate one for each model: 28-day model uses the card for 56 capsules, 56-day model uses the card for 112 capsules. But the task says packaging cost is "matched by blister_card_count" — so check if each medication has its own blister_card_count or if it's a lookup table.
- The `blister_card_count` field appears in the output schema per medication, suggesting each medication has a specific blister card count. This likely means the 28-day and 56-day models use the SAME card cost per fill for a given medication (the card count doesn't change between models — only the number of fills changes). Read the data carefully to determine the correct interpretation.
- Do NOT invent data. Use only what's in the CSV files.
- All currency values must be rounded to exactly 2 decimal places in the JSON output.

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