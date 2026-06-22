# Task Instruction

Complete the following task to update a Korean HWPX document.

## Goal
Revise `renewal_playbook.hwpx` using data from `renewal_update.json` and `followups.csv`, saving the result to `/root/renewal_playbook_updated.hwpx`.

## Step-by-step Plan

### Step 1: Explore the workspace
- List all files in the working directory to find `renewal_playbook.hwpx`, `renewal_update.json`, and `followups.csv`.
- Read `renewal_update.json` completely and note every field (customer name, current owner, renewal window, pricing band, escalation contact, pricing note, and any others).
- Read `followups.csv` completely. Note the `sequence` column and the content of each follow-up item.

### Step 2: Understand the HWPX package structure
- HWPX is a ZIP archive. Unzip `renewal_playbook.hwpx` into a temporary directory (e.g., `/tmp/hwpx_work/`).
- List all files in the extracted archive to understand the structure.
- Identify the main content XML file(s) — typically under `Contents/` and named something like `section0.xml` or `content.xml`.
- Read and display the full content of each XML content file to understand the document structure.

### Step 3: Identify what needs changing
- In the content XML, locate:
  a. All occurrences of the OLD customer name, current owner, renewal window, pricing band, escalation contact, and pricing note (the current/old values will be visible in the XML text nodes).
  b. The three existing follow-up lines that need replacement.
  c. The appendix sentence `이 부록 문단은 그대로 유지해야 합니다.` — confirm its location and mark it as DO NOT MODIFY.
- Cross-reference old values in the XML with the new values from `renewal_update.json`.

### Step 4: Write a Python script to perform the edits
Write a Python script that:
1. Extracts the HWPX ZIP to a temp directory.
2. Parses each content XML file using `xml.etree.ElementTree` (with proper namespace handling).
3. For each paragraph element in editable sections:
   a. Replaces all old field values with new values from `renewal_update.json` in text nodes.
   b. For follow-up lines: identifies the three old follow-up lines and replaces them with the CSV items sorted by `sequence` column. Removes old follow-up paragraphs and inserts new ones in their place (do NOT leave duplicates).
   c. For any paragraph whose text content was modified: removes any child elements that represent layout cache (look for elements like `linesegarray`, `lineSegArray`, `hp:linesegarray`, or similar layout-cache/segment elements). These cached layout elements cause overlapping characters if left stale.
   d. Does NOT modify the paragraph containing `이 부록 문단은 그대로 유지해야 합니다.`.
4. Writes the modified XML back (preserving XML declaration, namespaces, and encoding).
5. Re-packages everything into a new ZIP file at `/root/renewal_playbook_updated.hwpx`, preserving the original directory structure and using the same compression method.

### Step 5: Execute and validate
- Run the Python script.
- Verify the output exists at `/root/renewal_playbook_updated.hwpx`.
- Unzip the output to a verification directory and inspect the content XML to confirm:
  a. All old values are replaced with new values (grep for old values — should find none).
  b. New follow-up lines appear in correct sequence order.
  c. No duplicate/stale follow-up lines remain.
  d. The appendix sentence is preserved exactly.
  e. Layout cache elements are removed from modified paragraphs.
  f. The file is a valid ZIP.

## Critical Details
- Namespace handling: HWPX XML uses namespaces (e.g., `http://www.hancom.co.kr/hwpml/...`). Register all namespaces before parsing so they are preserved on output. Use `ET.register_namespace()` for each namespace found.
- When re-serializing XML, preserve the original encoding declaration (typically UTF-8).
- When creating the output ZIP, iterate over the original ZIP's entries to preserve the exact directory structure and entry names. Use `zipfile.ZIP_DEFLATED` compression.
- The layout cache element name varies by HWPX version. Look for any element whose local name contains `lineseg`, `lineSegArray`, `LineSeg`, or `charShapeArray` cache within paragraph (`<hp:p>`) elements. Inspect the actual XML to determine the exact element name before writing removal logic.
- For text replacement, be thorough: a single paragraph may contain multiple `<hp:run>` elements with text split across `<hp:t>` tags. Search and replace across all text nodes.
- When replacing follow-up lines, match them by their content (the old follow-up text), not by position index, to be robust.

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