# Task Instruction

Complete the HWPX project proposal document by following these steps precisely:

## Step 1: Inspect the input files
- Read `/root/project_proposal.json` to understand the placeholder values.
- List the contents of `project_proposal_template.hwpx` (it's a ZIP file) to understand its structure.
- Extract the HWPX to a temporary directory (e.g., `/tmp/hwpx_work/`).
- Find and read the XML content files (likely under `Contents/`) to locate all `{{...}}` placeholders.

## Step 2: Write a Python script to perform the transformation
Create a Python script that does the following:

### 2a: Load JSON data
- Parse `project_proposal.json` and build a mapping of placeholder names to values.

### 2b: Budget normalization
- For the budget value, remove commas but keep the leading currency symbol (e.g., `₩1,500,000` → `₩1500000`).

### 2c: Replace placeholders in XML files
- For each XML file in the extracted HWPX (especially section XML files under `Contents/`):
  - Read the raw XML as a string.
  - Use regex `\{\{[^}]*\}\}` to find placeholders. IMPORTANT: Placeholders may be split across multiple `<hp:t>` tags due to HWPX editor behavior. The regex must work across tag boundaries.
  - For each placeholder found, replace it with the corresponding JSON value inside a single `<hp:t>` tag.
  - Track which `<hp:p>` paragraphs were modified.

### 2d: Append month spans to phase lines
- For lines containing phase markers (단계1, 단계2, 단계3), parse the date range already present in that line.
- Calculate the month span using: `(end_year - start_year) * 12 + (end_month - start_month + 1)`
- Append ` (N개월)` after the phase content in the appropriate `<hp:t>` tag.
- Expected results based on typical project proposals: 단계1 → (3개월), 단계2 → (3개월), 단계3 → (1개월). But calculate from actual dates rather than hardcoding.

### 2e: Remove layout cache from modified paragraphs
- For every `<hp:p>` paragraph that was modified, remove any `<hp:lineSegArray>...</hp:lineSegArray>` elements using regex. This prevents overlapping character rendering artifacts.

### 2f: Verify no remaining placeholders
- After all replacements, scan the entire XML for any remaining `{{` patterns. If found, report and fix them.

## Step 3: Repackage as HWPX
- Write the modified XML files back to the extracted directory.
- From WITHIN the extracted directory root, run `zip -r /root/project_proposal_ready.hwpx .` to create the output file. This ensures the internal structure (Contents/, META-INF/, etc.) is at the ZIP root level, which is required for HWPX validity.

## Step 4: Validate
- Verify `/root/project_proposal_ready.hwpx` exists and is a valid ZIP.
- Unzip it to a verification directory and check:
  - No `{{` patterns remain in any XML file.
  - The budget value has no commas but retains the currency symbol.
  - Phase lines have the `(N개월)` annotations.
  - Korean labels and static note lines are unchanged.
  - No `<hp:lineSegArray>` elements exist in paragraphs that were modified.

## Important Notes
- Do NOT use an XML parser that might reformat or alter the XML structure. Use string/regex-based manipulation to preserve the exact XML formatting.
- Be careful with encoding: HWPX XML files are typically UTF-8.
- The HWPX file is a ZIP archive - use standard zip/unzip tools or Python's zipfile module.

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