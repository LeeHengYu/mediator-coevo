# Task Instruction

Complete the following task to update a renewal playbook HWPX document.

## Goal
Revise `renewal_playbook.hwpx` using data from `renewal_update.json` and `followups.csv`, saving the result to `/root/renewal_playbook_updated.hwpx`.

## Step-by-step Plan

### Step 1: Inspect all input files
- List the contents of the working directory to find all input files.
- `cat renewal_update.json` to see the update fields and their new values.
- `cat followups.csv` to see the follow-up items and their sequence column.
- Examine the HWPX file structure: `unzip -l renewal_playbook.hwpx` to list its contents.
- Extract the HWPX to a temp directory: `mkdir /tmp/hwpx_work && cp renewal_playbook.hwpx /tmp/hwpx_work/original.zip && cd /tmp/hwpx_work && unzip original.zip -d extracted`
- Read the main content XML files (likely `Contents/section0.xml` or similar) to understand the document structure, identify where the fields appear, find the three follow-up lines, and locate the appendix sentence.

### Step 2: Understand the data mapping
- From the JSON, identify the old→new mappings for: customer name, current owner, renewal window, pricing band, escalation contact, and pricing note.
- From the CSV, read follow-up items sorted by the `sequence` column.
- In the XML, identify the exact text content of the three existing follow-up lines that need replacement.

### Step 3: Write a Python script to perform the update
Create a Python script that:

1. **Extracts** the HWPX ZIP to a temporary directory.
2. **Reads** `renewal_update.json` and `followups.csv`.
3. **Finds all XML files** in the extracted package (especially section XML files in Contents/).
4. **For each XML file**, parses it and:
   a. Performs text replacements for all six fields (customer name, current owner, renewal window, pricing band, escalation contact, pricing note) — replace OLD values with NEW values everywhere they appear in text nodes.
   b. Identifies the three follow-up line paragraphs and replaces them with the CSV items in sequence order. If there are exactly 3 follow-up paragraphs and a different number of CSV items, handle accordingly (add or remove paragraph elements as needed).
   c. **Critically**: For every paragraph element whose text content was modified, remove all `linesegarray` child elements (layout cache). Use namespace-aware searching — the element local name is `linesegarray`. This prevents overlapping characters when the document is opened.
   d. Verifies the appendix sentence `이 부록 문단은 그대로 유지해야 합니다.` is present and unchanged.
5. **Writes** modified XML back to the extracted directory.
6. **Repackages** the directory as a valid ZIP/HWPX file at `/root/renewal_playbook_updated.hwpx`, preserving the original ZIP structure (mimetype file first if present, no extra compression on mimetype).

### Step 4: Important implementation details
- **HWPX namespace handling**: HWPX XML uses namespaces like `http://www.hancom.co.kr/hwpml/2011/paragraph` (hp namespace). Use `lxml` or `xml.etree.ElementTree` with proper namespace handling. When searching for `linesegarray`, match on local name: `if element.tag.endswith('}linesegarray')` or use XPath with `local-name()`.
- **Follow-up replacement strategy**: First, read the existing XML to identify the exact follow-up lines (they likely contain numbered items like "1. ...", "2. ...", "3. ..."). Find their parent paragraph elements. Replace the text content of these paragraphs with the CSV follow-up items in sequence order. If there are more or fewer CSV items than existing follow-up paragraphs, clone or remove paragraph elements as needed.
- **Text replacement must be thorough**: Search ALL text nodes (including `<hp:t>` or `<t>` elements within runs) for old values and replace with new values. The old values should be derived from the current document content by comparing with the JSON's new values — the JSON likely has keys indicating which field maps to which new value. Inspect the JSON structure carefully to determine old vs new values.
- **ZIP repackaging**: Use Python's `zipfile` module. If there's a `mimetype` file, add it first with `ZIP_STORED` (no compression). Add all other files with `ZIP_DEFLATED`.

### Step 5: Execute and validate
- Run the Python script.
- Verify the output: `unzip -l /root/renewal_playbook_updated.hwpx` to confirm valid ZIP.
- Extract and inspect the updated XML to confirm:
  - All six fields have been updated (no old values remain).
  - Follow-up lines match CSV items in sequence order.
  - Appendix sentence is preserved.
  - No `linesegarray` elements remain in modified paragraphs.
  - The XML is well-formed.

### Step 6: Final check
- Run any available test or verifier script if present in the task directory.
- Confirm `/root/renewal_playbook_updated.hwpx` exists and is a valid ZIP file.

## Critical Reminders
- Do NOT skip the linesegarray removal step — this is essential for the verifier.
- Do NOT modify the appendix sentence.
- Do NOT leave old values alongside new values — replace, don't append.
- Inspect the actual file contents before writing code — do not assume structure.
- After editing, re-read the output XML to confirm changes landed correctly.

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