# Task Instruction

## Task: Update renewal_playbook.hwpx with new data

You need to revise `renewal_playbook.hwpx` using `renewal_update.json` and `followups.csv`, saving the result to `/root/renewal_playbook_updated.hwpx`.

### Step 1: Inspect available files and understand the data

1. List files in the task directory to find `renewal_playbook.hwpx`, `renewal_update.json`, and `followups.csv`.
2. Read `renewal_update.json` completely — note every field (customer name, current owner, renewal window, pricing band, escalation contact, pricing note) and both old and new values.
3. Read `followups.csv` completely — note the columns and sort by `sequence` column to determine the correct order.

### Step 2: Explore the .hwpx package structure

1. A `.hwpx` file is a ZIP archive. Use Python to list all entries in the ZIP.
2. Extract and inspect the XML files inside, particularly the main content XML (likely under `Contents/` — look for files like `section0.xml` or similar content XML files).
3. Identify the XML namespace(s) used.
4. Find all editable text content — look for text elements, paragraph elements, and how text is structured (it may be split across multiple `<hp:t>` or similar run elements within a paragraph).
5. Identify the three existing follow-up lines in the content.
6. Locate the appendix sentence `이 부록 문단은 그대로 유지해야 합니다.` and note its exact position.
7. Identify any layout-cache elements (look for elements like `<hp:linesegarray>`, `<hp:lineSegArray>`, `<hp:lineseg>`, or similar caching/layout elements within paragraph structures).

### Step 3: Write a Python script to perform the update

Write a Python script that:

1. **Copies the .hwpx ZIP** to the output path, modifying only the content XML file(s).
2. **Reads the JSON** to get old→new value mappings for all six fields.
3. **Reads the CSV** and sorts by `sequence` to get ordered follow-up items.
4. **Parses the content XML** using `lxml.etree` (preferred) or `xml.etree.ElementTree` with proper namespace handling.
5. **Performs text replacements** for all six fields:
   - For each field, find every occurrence of the old value and replace with the new value.
   - IMPORTANT: Text may be split across multiple run elements within a single paragraph. You must handle this by either:
     (a) Concatenating text from all runs in a paragraph, checking for matches, and if found, reconstructing the runs with the new text, OR
     (b) Checking individual text elements for partial or complete matches.
   - The safest approach: for each paragraph, collect all text content, perform replacements on the concatenated string, then redistribute the text back (e.g., put all text in the first run and clear the rest, or merge runs).
6. **Replaces follow-up lines**: Identify the three existing follow-up paragraphs (they likely contain numbered or sequential follow-up text). Replace their text content with the CSV items in `sequence` order. If there are exactly 3 old follow-up lines and a different number of CSV items, add or remove paragraph elements as needed.
7. **Removes layout-cache elements** from any paragraph whose text was modified. Look for elements like `linesegarray`, `lineSegArray`, `lineseg`, or any element that appears to be a layout/rendering cache within `<hp:p>` or paragraph elements. Remove these elements entirely from modified paragraphs only.
8. **Preserves the appendix sentence** `이 부록 문단은 그대로 유지해야 합니다.` — do NOT modify the paragraph containing this text.
9. **Writes the modified XML** back into the ZIP package, preserving all other ZIP entries exactly as-is.

### Step 4: Validate the output

1. Verify `/root/renewal_playbook_updated.hwpx` exists and is a valid ZIP file.
2. Extract and inspect the modified content XML to confirm:
   - All six fields have been updated (old values no longer appear, new values are present).
   - The follow-up lines match the CSV data in sequence order.
   - The appendix sentence is unchanged.
   - No layout-cache elements remain in modified paragraphs.
   - The XML is well-formed.
3. List the ZIP contents to ensure all original entries are preserved.

### Critical Details

- **Do not use string replacement on raw XML** — always parse and modify the DOM tree to avoid breaking XML structure.
- **Namespace handling**: HWPX uses namespaces like `http://www.hancom.co.kr/hwpml/2011/paragraph` or similar. Register and use them properly in XPath queries.
- **When replacing text across split runs**: If a paragraph has text split like `<t>Cust</t><t>omer ABC</t>`, a simple per-element replacement won't catch `Customer ABC`. Concatenate, replace, then put the result back.
- **Layout cache removal**: Only remove from paragraphs you actually modified. Do not touch unmodified paragraphs.
- **ZIP reconstruction**: Use `zipfile.ZipFile` to read the original and write the new file. For each entry, if it's the content XML, write the modified version; otherwise, copy the original bytes exactly.
- **Encoding**: Write XML with the same encoding declaration as the original (likely UTF-8).
- **Remove old values completely** — no duplicate lines should remain.

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