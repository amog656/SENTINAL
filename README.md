# DETECTIVE
# AI-Powered Incident Investigation Platform

An intelligent investigation system designed to uncover the truth hidden across fragmented operational knowledge.

Modern organizations generate vast amounts of information every day through incident reports, deployment notes, architecture documents, troubleshooting guides, postmortems, engineering discussions, customer complaints, and internal documentation. Over time, this knowledge becomes scattered across systems, teams, and formats, making it increasingly difficult to answer critical operational questions.

When an engineer asks:

> "Why did the Orders API become slow after yesterday's deployment?"

the answer rarely exists in a single document.

The relevant evidence may be distributed across multiple incident reports, deployment records, previous outages, architecture changes, troubleshooting procedures, and historical investigations. Some documents may contain outdated recommendations. Others may contradict each other. Important information may only emerge after connecting evidence from several independent sources.

This platform is designed to solve that problem.

Rather than acting as a traditional search engine, the system functions as an investigation agent capable of gathering evidence, performing iterative research, reconstructing timelines, identifying contradictions, and generating evidence-backed conclusions.

The platform accepts natural-language questions and launches a structured investigation workflow. It searches across multiple document types, discovers relevant evidence, performs additional searches based on newly uncovered information, analyzes relationships between events, and builds a comprehensive understanding of the issue before presenting findings.

The system is designed around the principle that answers should be explainable.

Every conclusion is supported by traceable evidence, investigation paths, confidence estimates, and supporting documentation. Instead of returning a simple answer, the platform demonstrates how that answer was reached.

---

## Core Capabilities

### Natural Language Investigation

Users can interact with the platform using ordinary language.

Examples:

- Why did the Orders API become slow after deployment?
- Have we experienced this incident before?
- What changed between the previous stable release and the current deployment?
- Which document contains the most recent recommendation for this issue?
- What is the likely root cause of the current outage?

The platform converts these questions into structured investigations and automatically begins evidence collection.

---

### Multi-Hop Research and Investigation

Complex questions often require multiple rounds of research.

The platform continuously expands its investigation using information discovered during previous steps.

Example:

```text
User Question
        ↓
Initial Search
        ↓
Deployment Record Found
        ↓
Extract Service Version
        ↓
Search Previous Incidents Using Same Version
        ↓
Find Similar Failures
        ↓
Locate Postmortems
        ↓
Compare Root Causes
        ↓
Generate Findings
