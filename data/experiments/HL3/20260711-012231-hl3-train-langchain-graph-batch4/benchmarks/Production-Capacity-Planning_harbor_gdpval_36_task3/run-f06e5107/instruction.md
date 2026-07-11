# Task Instruction

Create a Python script `/root/solve.py` and execute it to produce `/root/assembly_plan.xlsx` and `/root/assembly_summary.txt`.

The script must do the following:

### Step 1: Read input data
- Open `/root/assembly_schedule.xlsx`, sheet `Assembly`.
- Find the column labeled `PCB Assembly Demand (Std Hrs)` and the column that identifies phases.
- Extract phases 6 through 54 inclusive. If there are duplicate phase entries, keep only the first occurrence of each phase.
- Store the scheduled demand for each phase.

### Step 2: Simulate the deterministic planning policy

Initialize:
- For Phase 6: `Calc Start` is derived from the initial condition: `Start of Phase Past Due + Scheduled Demand = 469.59`, so `Calc Start = 469.59 - Scheduled_Demand_Phase6`. The reported `Start of Phase Past Due` for Phase 6 is `max(0, Calc Start)`.

For each phase (6..54) in order:
1. `Start of Phase Past Due (Std Hrs) = max(0, Calc_Start)` (for reporting/display)
2. Choose `Days Worked`:
   - If `Start of Phase Past Due > 0.01` (i.e., `max(0, Calc_Start) > 0.01`):
     - Try 5 first: if `Calc_Start + Scheduled_Demand - (20 * 5) <= 0`, pick 5.
     - Else try 6: if `Calc_Start + Scheduled_Demand - (20 * 6) <= 0`, pick 6.
     - Else pick 6 anyway.
   - Else (past due <= 0.01):
     - If `Scheduled_Demand <= 80`, pick 4. Else pick 5.
3. `Weekly Capacity (Std Hrs) = 20 * Days_Worked`
4. `End of Phase Backlog/Buffer (Std Hrs) = Calc_Start + Scheduled_Demand - Weekly_Capacity`
5. `Overtime Hours = 10 * max(0, Days_Worked - 4)`
6. For the next phase: `Calc_Start = End of Phase Backlog/Buffer` (the signed value, not clamped).

### Step 3: Write `/root/assembly_plan.xlsx`
- Create a single worksheet named `Plan`.
- Row 1 headers exactly: `Phase`, `Days Worked`, `Scheduled Demand (Std Hrs)`, `Weekly Capacity (Std Hrs)`, `Start of Phase Past Due (Std Hrs)`, `End of Phase Backlog/Buffer (Std Hrs)`, `Overtime Hours`
- 49 data rows (phases 6..54), ascending order, no gaps, no duplicates.
- Use openpyxl to write the file.

### Step 4: Write `/root/assembly_summary.txt`
- Scan results to find:
  - `First_Week_5_Days`: the first phase where Days Worked drops to exactly 5 after having been 6 at some earlier phase. If Days Worked is never 6, or never drops to 5 after being 6, use `N/A`. ACTUALLY — re-read the spec: `First_Week_5_Days` is simply the first phase where Days Worked == 5. If none, `N/A`.
  - `First_Week_4_Days`: the first phase where Days Worked == 4. If none, `N/A`.
- Write exactly 3 lines:
  ```
  First_Week_5_Days: <phase-number-or-N/A>
  First_Week_4_Days: <phase-number-or-N/A>
  Summary: <manager-facing summary, ≤60 words, ≤3 sentences, mentioning both step-down phase numbers or N/A>
  ```

### Step 5: Validation
- After generating both files, re-read `/root/assembly_plan.xlsx` and print:
  - Number of data rows (should be 49)
  - The headers
  - First 5 rows and last 3 rows of data
  - Phase 6 values to verify initial condition: `Start of Phase Past Due + Scheduled Demand` should equal 469.59
- Re-read `/root/assembly_summary.txt` and print its contents.
- Count words in the Summary line to confirm ≤60.

### Important notes
- Inspect the input file first to identify the exact column names and phase column before processing. Print column names and first few rows.
- Use `pandas` for reading and `openpyxl` for writing the output Excel.
- Round all floating point values to 2 decimal places in the output.
- The `First_Week_5_Days` and `First_Week_4_Days` refer to the step-down transitions in the context of the catch-up plan — they represent when capacity requirements ease. `First_Week_5_Days` = first phase with exactly 5 days worked; `First_Week_4_Days` = first phase with exactly 4 days worked.

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
Task metadata: author_email=codex@openai.com, author_name=Codex, category=manufacturing-planning, difficulty=medium, tags=[xlsx, operations, capacity-planning, pcb, backlog].
Verifier config: timeout_sec=900.0.