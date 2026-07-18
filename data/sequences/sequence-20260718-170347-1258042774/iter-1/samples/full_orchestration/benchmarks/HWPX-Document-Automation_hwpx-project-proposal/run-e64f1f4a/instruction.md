# Task Instruction

Create a Python script and execute it to complete the project proposal document. The script should:

1. Read the JSON file at `/root/project_proposal.json` to get the replacement values.
2. Read the HWPX template at `/root/project_proposal_template.hwpx` (which is a ZIP/OCF package).
3. For each XML file inside the ZIP (especially `Contents/section0.xml` or similar section files):
   a. Parse the XML content as text.
   b. Replace all `{{...}}` placeholders with corresponding values from the JSON. Match placeholder names to JSON keys (e.g., `{{project_name}}` -> JSON key `project_name`).
   c. Normalize the budget value: remove commas but keep the leading currency symbol (e.g., `₩1,000,000` -> `₩1000000`).
   d. For phase lines containing `단계1`, `단계2`, `단계3`, append a parenthesized month span calculated from the date range in that line:
      - Parse the start and end dates from the line (likely in YYYY.MM.DD or similar format).
      - Calculate the month difference.
      - Append ` (N개월)` after the phase content on that line.
   e. After modifying paragraph text, remove layout-cache elements (such as `<hp:lineSegArray>`, `<hp:lineseg>`, `<hp:lineSegArray>...</hp:lineSegArray>` blocks, and similar `lineSegArray` or `lineSeg` elements) from any `<hp:p>` paragraph that was modified. Use XML parsing (lxml or ElementTree with namespace awareness) or regex to strip these elements.
4. Ensure no `{{...}}` placeholder text remains anywhere in any file in the output package.
5. Write the resulting HWPX package to `/root/project_proposal_ready.hwpx` maintaining the same ZIP structure.

Detailed steps:
- First, inspect the template HWPX to list all files inside it and examine the section XML to understand the structure, placeholder names, and phase line formats.
- Then inspect the JSON file to see available keys and values.
- Write and run the Python script based on what you find.
- After running, verify:
  - The output file exists at `/root/project_proposal_ready.hwpx`
  - It is a valid ZIP file
  - No `{{` or `}}` remains in any XML content within the ZIP
  - Phase lines have the month span appended
  - Budget value has no commas
  - Layout-cache elements are removed from modified paragraphs
  - Korean labels and static note lines are unchanged

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
Task metadata: author_email=catpaw@example.com, author_name=CatPaw Task Engineer, category=document-editing, difficulty=medium, tags=[hwpx, xml-editing, document-processing, latent-method-reuse].
Verifier config: timeout_sec=600.0.