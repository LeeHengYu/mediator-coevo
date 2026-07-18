# Task Instruction

You must produce the file `/root/renewal_playbook_updated.hwpx` by programmatically editing the existing HWPX package. Follow every step below precisely.

## Background
A `.hwpx` file is a ZIP archive containing XML files (ODF-like structure). The editable text lives inside XML files under `Contents/` (typically `section0.xml` or similar). Layout-cache elements such as `<hp:lineSegArray>`, `<hp:lineSeg>`, `<hp:linesegarray>`, etc., inside any `<hp:p>` paragraph whose text you modify must be removed so the document opens cleanly.

## Step-by-step

### 1. Inspect the inputs
- Read `/root/renewal_update.json` and print its contents.
- Read `/root/followups.csv` and print its contents.
- List the files inside `/root/renewal_playbook.hwpx` (it is a ZIP). Identify which XML file(s) under `Contents/` contain the editable body text. Print the full XML of each such file so you can see every paragraph, tag, and layout-cache element.

### 2. Understand the update data
- `renewal_update.json` contains key-value pairs for fields like customer name, current owner, renewal window, pricing band, escalation contact, pricing note (or similarly named keys). Each old value in the document must be replaced with the corresponding new value.
- `followups.csv` contains follow-up items with a `sequence` column (or similar ordering column). These items must replace the existing three follow-up lines in the document, ordered by `sequence`.

### 3. Write a Python script that does the following

#### a. Copy the HWPX ZIP
Copy `/root/renewal_playbook.hwpx` to `/root/renewal_playbook_updated.hwpx`.

#### b. Open the copy as a ZIP (in-place update)
Use `zipfile.ZipFile` to read all entries. For each XML file under `Contents/`:

#### c. Parse the XML with `lxml.etree` (or `xml.etree.ElementTree`)
- Use the parser that preserves namespaces.

#### d. Perform field replacements
For every field in `renewal_update.json`:
- Identify the OLD value and NEW value.
- Walk every text node in the XML (element `.text` and `.tail`). Wherever the old value appears as a substring, replace it with the new value.
- **Important**: The JSON structure may nest old/new under each field key. Inspect the actual JSON structure and adapt accordingly.

#### e. Replace follow-up lines
- Identify the existing follow-up paragraphs. They are three consecutive paragraphs whose text matches a pattern (e.g., numbered follow-up items like `1차: ...`, `2차: ...`, `3차: ...` or similar). Print them so you can confirm.
- Replace their text content with the CSV rows sorted by `sequence`. Match the count: if CSV has 3 rows, replace the 3 existing paragraphs. If the CSV has more or fewer rows, add or remove `<hp:p>` elements accordingly (clone structure from an existing follow-up paragraph).
- The replacement text for each paragraph should follow the same formatting pattern as the originals (e.g., `1차: <text>` becomes the new text from CSV in sequence order).

#### f. Remove layout-cache from modified paragraphs
For every `<hp:p>` element whose text content was changed:
- Remove all child elements that are layout-cache related: `lineSegArray`, `lineSeg`, `linesegarray`, `lineseg` (case-insensitive local name matching). Use namespace-aware or local-name matching.
- This is critical to prevent overlapping-character corruption.

#### g. Verify the appendix sentence
Confirm that the string `이 부록 문단은 그대로 유지해야 합니다.` still exists unchanged in the XML after all edits. If it was accidentally modified, restore it. Print confirmation.

#### h. Write back into the ZIP
Rewrite the modified XML files back into the ZIP archive at their original paths. Preserve all other files (images, settings, etc.) unchanged.

### 4. Validate the output
- Open `/root/renewal_playbook_updated.hwpx` as a ZIP and confirm it is valid.
- Read the modified XML file(s) and print them to verify:
  - All old field values are gone (search for each old value and confirm zero occurrences).
  - All new field values are present.
  - Follow-up lines match the CSV data in sequence order.
  - The appendix sentence is intact.
  - No layout-cache elements remain in modified paragraphs.
  - The overall XML is well-formed.

### 5. Important constraints
- Do NOT add duplicate lines; remove old values before inserting new ones.
- Do NOT modify the appendix sentence `이 부록 문단은 그대로 유지해야 합니다.`.
- The output must be a valid ZIP (`.hwpx` package).
- Work carefully with namespaces in the XML; HWPX uses `hp:` and other namespace prefixes.

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