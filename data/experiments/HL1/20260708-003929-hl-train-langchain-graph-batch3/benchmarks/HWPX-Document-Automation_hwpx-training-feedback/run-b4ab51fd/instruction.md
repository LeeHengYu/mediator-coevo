# Task Instruction

Complete the following task step by step.

## Goal
Fill in the training feedback sheet `training_feedback_template.hwpx` using values from `training_feedback.json`, and save the result to `/root/training_feedback_ready.hwpx`.

## Steps

### Step 1: Inspect the JSON data
Read and display the contents of `training_feedback.json` in the task directory. Note all keys and values.

### Step 2: Inspect the HWPX package structure
A `.hwpx` file is a ZIP archive. List all files inside `training_feedback_template.hwpx` to understand its structure. Then read each XML file inside the archive (especially files under `Contents/` such as `section0.xml`, `content.hpf`, `header.xml`, etc.) and identify:
- All `{{...}}` placeholders and which XML files they appear in
- The XML namespace declarations used
- Any layout-cache elements (look for elements like `<hp:linesegarray>`, `<hp:lineSegArray>`, `<lineseg>`, `<hp:LineSeg>`, or similar glyph/character positioning cache elements within paragraph elements)

Print the full content of each XML file that contains placeholders.

### Step 3: Write and execute a Python script
Write a Python script that:

1. **Loads the JSON** data from `training_feedback.json`.

2. **Copies the HWPX template** to `/root/training_feedback_ready.hwpx` first, then operates on the copy (or builds a new ZIP from the template's contents).

3. **For each file in the ZIP archive**, reads the content. For XML files that contain `{{` placeholders:

   a. **Replace `{{참석자수}}`** (or whatever the exact placeholder name is for attendee count) with digits only. If the JSON value is like `25명` or `25`, extract just the numeric digits.
   
   b. **Replace `{{만족도}}`** (or the satisfaction score placeholder) with the format `X.X점 (5.0점 만점)` where X.X is the numeric score from the JSON.
   
   c. **Replace the overall opinion placeholder** with the JSON value, then append ` 후속 심화반 검토 요망.` after it. Make sure there's a space before the appended text if the original doesn't end with one.
   
   d. **Replace all other `{{...}}` placeholders** with their corresponding JSON values directly.

4. **Remove stale layout-cache elements** from any paragraph whose text was modified. In HWPX XML, paragraphs typically have a structure like `<hp:p>` containing child elements. Look for and remove elements that represent layout caches. Common element names include:
   - `<hp:linesegarray>` or `<hp:lineSegArray>` (and their children)
   - Any element whose local name contains `LineSeg` or `lineseg`
   
   Use an XML parser (like `lxml.etree` or `xml.etree.ElementTree`) with proper namespace handling to parse, modify, and serialize the XML. Be careful to preserve namespace prefixes and declarations.

5. **Write the modified ZIP** ensuring all original entries are preserved (including non-XML files like images, settings, etc.) with their original compression settings.

6. **Validate** the output:
   - Confirm `/root/training_feedback_ready.hwpx` is a valid ZIP
   - Confirm no `{{` or `}}` patterns remain in any file within the archive
   - Print the text content of modified XML files to verify correctness

### Step 4: Final verification
Open the output HWPX as a ZIP and grep all files for any remaining `{{` patterns. Print a summary of all replacements made. Confirm the file exists at `/root/training_feedback_ready.hwpx`.

## Important Notes
- Use `lxml.etree` if available, otherwise `xml.etree.ElementTree` for XML parsing. Namespace-aware parsing is critical.
- When serializing XML back, preserve the XML declaration and encoding.
- The HWPX format uses namespaces heavily. When searching for layout cache elements, use namespace-aware queries or iterate all elements checking local names.
- Do NOT use simple string replacement for XML modification if it risks breaking XML structure. Use an XML parser for structural changes (like removing cache elements) but string replacement is acceptable for simple text substitutions within text nodes if done carefully.
- Keep all Korean labels and static note lines unchanged - only replace placeholder text.
- Ensure the final file has no remaining `{{...}}` placeholder text anywhere.

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