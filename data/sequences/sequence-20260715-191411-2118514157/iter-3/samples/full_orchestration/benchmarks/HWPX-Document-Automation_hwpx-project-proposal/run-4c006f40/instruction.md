# Task Instruction

Create and run a single Python script that:

1. **Understand the HWPX format**: A `.hwpx` file is a ZIP archive containing XML files. The main document content is typically in `Contents/section0.xml` (or similar path). Inspect the archive to find all XML files.

2. **Read the JSON data**: Load `project_proposal.json` from the task directory to get all replacement values.

3. **Inspect the template**: Extract and print the content XML file(s) from `project_proposal_template.hwpx` to understand:
   - The XML namespace structure
   - Where `{{...}}` placeholders appear
   - The structure of phase lines (단계1, 단계2, 단계3) and their date ranges
   - Where budget values appear
   - What layout-cache elements exist (look for `linesegarray` or similar elements)

4. **Process the document** with these specific rules:
   a. **Replace all `{{...}}` placeholders** with corresponding values from the JSON file. Match placeholder names to JSON keys.
   b. **Normalize the budget value**: Remove commas from the budget number but keep the leading currency symbol (e.g., `₩1,000,000` → `₩1000000`).
   c. **Append month spans to phase lines**: For each phase line (단계1, 단계2, 단계3), calculate the month difference from the date range already present in that line, then append a parenthesized month span:
      - 단계1 → append `(3개월)`
      - 단계2 → append `(3개월)`  
      - 단계3 → append `(1개월)`
      Important: Actually compute the month spans from the dates in the lines to verify these expected values. If the dates differ, compute accordingly.
   d. **Keep all Korean labels and static note lines unchanged.**
   e. **Remove layout-cache elements** (`<linesegarray>...</linesegarray>` or any element named `linesegarray` in whatever namespace) from every paragraph whose text content you modified. This is critical for the document to render correctly.

5. **Rebuild the HWPX package**: Create the output at `/root/project_proposal_ready.hwpx` by copying all entries from the original ZIP, replacing only the modified XML file(s). Preserve all other entries exactly as-is.

6. **Validate the output**:
   - Open the output file as a ZIP and verify it's valid.
   - Re-read the modified XML and confirm:
     - No `{{` or `}}` patterns remain anywhere in the text content.
     - Phase lines contain the parenthesized month spans.
     - Budget value has no commas but retains the currency symbol.
     - Korean labels are preserved.
     - No `linesegarray` elements exist in modified paragraphs.
   - Print a summary of all replacements made and validation results.

**Implementation approach**: Use Python's `zipfile` module for archive handling and `lxml.etree` (preferred) or `xml.etree.ElementTree` for XML parsing. Do NOT use string replacement on raw XML—parse the XML properly to avoid breaking tags. When replacing text in `<t>` elements (or whatever tag holds text), iterate through all text-bearing elements and apply replacements.

**Important**: First do an exploratory pass—print the ZIP file listing and the raw XML content of the section file(s)—before writing the transformation logic. This ensures you understand the exact structure. Then implement the full transformation in the same script.

The template file is at: `/root/project_proposal_template.hwpx` (or check the current working directory if not found at /root).
The JSON file is at: `/root/project_proposal.json` (or check the current working directory).
Output must be saved to: `/root/project_proposal_ready.hwpx`

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