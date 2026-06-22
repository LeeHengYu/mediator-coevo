# Task Instruction

Execute the following steps in order:

1. **Inspect the input files** to understand their structure and contents:
 ```
 cat /root/ingredient_cost.csv
 cat /root/card_cost.csv
 cat /root/reimbursement.csv
 ```

2. **Write and run a Python script** (`/root/solve.py`) that performs the full analysis. The script must:

 a. **Read the three CSV files** using the `csv` module (or pandas if available).
 
 b. **Merge/join** the data by medication name so that for each medication you have:
 - `medication` (string)
 - `price_per_1000_capsules_usd` (from `ingredient_cost.csv`)
 - `blister_card_count` (from `ingredient_cost.csv` or `card_cost.csv` — inspect to see which file has it)
 - `card_cost_usd` (from `card_cost.csv`, matched by `blister_card_count`)
 - `reimbursement_per_cycle_180_patients_usd` (from `reimbursement.csv`)

 c. **For each medication, compute** (all floats, rounded to 2 decimals at the end):

 - **Annual drug cost (28-day):** `(price_per_1000_capsules_usd / 1000) * 56 * 180 * 12`
 - **Annual drug cost (56-day):** `(price_per_1000_capsules_usd / 1000) * 112 * 180 * 6`
 - Note: both should be identical since 56×12 = 112×6 = 672 capsules/patient/year × 180 patients.
 - **Annual packaging cost (28-day):** `card_cost_usd * 180 * 12`
 - **Annual packaging cost (56-day):** `card_cost_usd * 180 * 6`
 - **Annual reimbursement (28-day):** `reimbursement_per_cycle_180_patients_usd * 12`
 - **Annual reimbursement (56-day):** `reimbursement_per_cycle_180_patients_usd * 6`
 - **Annual margin (28-day):** `annual_reimbursement_28 - annual_drug_cost_28 - annual_packaging_cost_28`
 - **Annual margin (56-day):** `annual_reimbursement_56 - annual_drug_cost_56 - annual_packaging_cost_56`
 - **Margin difference:** `annual_margin_56 - annual_margin_28`

 d. **Sort** the medications list alphabetically by `medication` (case-insensitive sort to be safe, but check actual data).

 e. **Compute totals:**
 - `total_annual_margin_28_day_usd` = sum of all 28-day margins
 - `total_annual_margin_56_day_usd` = sum of all 56-day margins
 - `total_annual_margin_difference_56_minus_28_usd` = sum of all per-medication differences
 - `absolute_total_margin_difference_usd` = abs(total_difference)

 f. **Decision rule:**
 - If `absolute_total_margin_difference_usd < 9000` → `"convert_to_56_day"`
 - Otherwise → `"keep_28_day"`
 - Write a short justification string that mentions the absolute difference and the threshold.

 g. **Round all currency values to 2 decimal places** before writing.

 h. **Write `/root/syncpack_analysis.json`** with the exact schema shown in the task. Use `json.dump` with `indent=2`. Ensure the key names match exactly (e.g., `annual_margin_difference_56_minus_28_usd`, not some variant).

 i. **Write `/root/syncpack_summary.md`** with 4–8 non-empty lines that include:
 - Total 28-day margin in USD
 - Total 56-day margin in USD
 - Absolute difference in USD
 - The exact decision slug (`convert_to_56_day` or `keep_28_day`)

3. **Run the script:**
 ```
 python3 /root/solve.py
 ```

4. **Validate the outputs:**
 - `cat /root/syncpack_analysis.json` — confirm it parses as valid JSON, has the `assumptions`, `medications` (sorted alphabetically), `totals`, and `recommendation` sections with all required keys.
 - `cat /root/syncpack_summary.md` — confirm 4–8 non-empty lines containing the required values and decision slug.
 - Verify that all currency values are rounded to exactly 2 decimal places.
 - Verify medications array is sorted alphabetically by medication name.
 - Verify the decision logic: if abs(total_difference) < 9000 then convert_to_56_day, else keep_28_day.

**Important details:**
- The `card_cost_usd` lookup: each medication has a `blister_card_count` (likely in `ingredient_cost.csv` or `reimbursement.csv`). The `card_cost.csv` maps `blister_card_count` → `card_cost_usd`. You must join on `blister_card_count` to get the per-fill packaging cost.
- Packaging cost is per patient per fill (i.e., each of 180 patients gets one card per fill, so annual = card_cost × 180 × fills_per_year).
- The drug cost formula: capsules_per_fill × fills_per_year × patients × (price_per_1000 / 1000).
- Do NOT invent data. Use only what is in the CSV files.
- If any CSV column names differ slightly from what's described, adapt accordingly but keep the JSON output keys exactly as specified.

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