# Comparing Local LLMs and NLP Methods for Fusing Text and Event Logs for Process Mining

Code, data and results of the paper. The pipeline classifies worker error reports into error classes, extracts event attributes (activity, machine, worker, time) and matches each report to its event in the event log with the 3-Level-Matching algorithm. Locally deployed LLMs (1B to 120B parameters, via Ollama) are compared with NLP baselines (TF-IDF, CRF, DistilBERT/BERT).

## Abstract

Process mining allows the discovery of business processes from event logs that have been recorded by traces of information systems. Usually, process mining techniques rely on the structured parts of these logs, such as case id, activity name, and timestamp. However, valuable contextual information of event logs is often embedded in unstructured textual data, which largely remains untapped. For example, while an event log may record the occurrence of an error, additional information is frequently captured in human-written error reports. This textual information can provide valuable insights. This paper presents an LLM-based pipeline for fusing error reports with event logs. Firstly, it classifies each error report into predefined error classes. Secondly, it extracts and harmonizes event log attributes (like timestamps, resources) from the reports and matches them to the event log based upon our 3-Level-Matching algorithm. The first level of the algorithm performs filtering, the second level applies embedding-based cosine similarity, while the final level relies on LLM support. The evaluation assesses the accuracy of the pipeline when using locally deployed LLMs of different sizes or various NLP baselines for classification and extraction components. For this, we extended the well-known event log from 4TU with worker style error reports. The results indicate high accuracy when using locally deployed LLMs. Specifically, GPT-OSS 20B achieves the best balance between hardware requirements and accuracy for our pipeline.

## Repository structure

| Path | Content |
|---|---|
| `Code/Config.py` | All settings: models, prompts, baselines, matching parameters, report generation |
| `Code/pipeline/` | Classification, extraction, NLP baselines, 3-Level-Matching, result tables and checks |
| `Code/generation/` | Generator of the synthetic error reports and the review harness |
| `Code/paper/` | Tables and figures of the paper |
| `Code/prompts/` | Classification and extraction prompts with the few-shot examples |
| `InputData/Raw/` | Production event log: Levy, D. (2014). *Production Analysis with Process Mining Technology*. 4TU.ResearchData. [https://doi.org/10.4121/uuid:68726926-5ac5-4fab-b873-ee76ea412399](https://doi.org/10.4121/uuid:68726926-5ac5-4fab-b873-ee76ea412399) |
| `InputData/Prepared/` | Prepared event log and candidate events for the error reports |
| `InputData/Reports.jsonl` | The 1,200 error reports with error class, difficulty level, ground truth and generation prompt |
| `InputData/Splits/` | Fixed split into 480 training, 120 validation and 600 evaluation reports |
| `Models/` | BERT base models for the NLP baselines (Zenodo release) |
| `Results/` | Results of the ten runs and the paper outputs |

`Models/` and the per-run folders `Results/run_XX/` are not stored in this repository because of their size. They are part of the Zenodo release: **[Zenodo DOI – to be added]**. Copy them into the repository folder to use them.

## Results

For each of the ten runs:

| File | Content |
|---|---|
| `Results/run_XX.csv` | One row per method, task, prompt, difficulty level and matching model: accuracy, invalid answers, timing |
| `Results/run_XX/predictions.csv` | Every classification and extraction prediction on the 600 evaluation reports: raw model answer, parsed prediction, correctness (overall and per attribute), model options, timing |
| `Results/run_XX/matching.csv` | Every matching decision: matched event, reference event, correctness, matching level, candidates and scores |
| `Results/run_XX/models.csv` | Ollama models with digest, parameter size and quantization |
| `Results/run_XX/training.csv` | Training settings, seed and selected epoch or hyperparameters of the baselines |

`Results/Summary.csv` summarizes all runs: one row per component, method, prompt, training size, difficulty level and metric, with the value of each run, the mean and the standard deviation. It covers classification (accuracy, macro-F1, number of reports predicted per class, including Other), extraction (overall and per attribute), matching (accuracy, assigned, wrongly assigned and unassigned reports, assignments per matching level, incorrect extractions matched correctly), the complete pipeline for all combinations of classification and extraction/matching methods, and the runtime per report.

`Results/Paper/` contains the tables and figures of the paper, generated from these files. Cells with lists or objects are stored as JSON. The settings of all runs are those in `Code/Config.py`.
## Installation

Python 3.13 or 3.14:

```sh
pip install -r requirements.txt
```

## Run the experiment

### Ollama

Install Ollama and pull the models:

```sh
ollama pull llama3.2:1b
ollama pull llama3.2:3b
ollama pull gemma3:4b
ollama pull ornith-1.5:9b
ollama pull gemma3:12b
ollama pull gemma4:e4b
ollama pull gpt-oss:20b
ollama pull gemma3:27b
ollama pull gemma4:31b
ollama pull nemotron3:33b
ollama pull llama3.1:70b
ollama pull gpt-oss:120b
ollama pull nemotron-3-super:120b
ollama pull mistral-medium-3.5      # report generation only
ollama pull mxbai-embed-large       # embeddings for the matching
```

The code connects to `http://127.0.0.1:11434`. For another address, set `OLLAMA_HOST`:

```sh
export OLLAMA_HOST=http://<host>:<port>
```

### BERT models

The NLP baselines use the unchanged base models `bert-base-uncased` and `distilbert-base-uncased` from Hugging Face (Apache 2.0) in `Models/bert-base-uncased` and `Models/distilbert-base-uncased`. The versions used for all runs are included in the Zenodo release. The fine-tuned models are trained again in every run and are not stored.

### Scripts

```sh
python Code/MainPrepareData.py       # prepares the event log and the candidate events
python Code/MainGenerateReports.py   # optional: generates the error reports again
python Code/MainBenchmark.py         # classification, extraction and matching of the next run
python Code/MainMatching.py          # optional: resumes the matching separately
python Code/MainResults.py           # writes the result files of runs that are still in progress
python Code/MainPaperResults.py      # tables and figures of the paper
python Code/MainSummary.py           # summary of all runs in Results/Summary.csv
```

Each start of `MainBenchmark.py` computes one complete run and numbers it after the existing runs; to start from `run_01`, move the existing runs out of `Results/`. An interrupted run is resumed by starting the script again: until a run is complete, its intermediate data is kept in `Results/Arbeitsdaten/run_XX`. When benchmark and matching are complete, the result files are written, read back and checked, and the intermediate data is removed (`ARBEITSDATEN_BEHALTEN = True` in `Code/Config.py` keeps it). LLM answers and the training of the baselines can differ slightly between hardware and Ollama versions.

## Prompts

- Classification and extraction: `Code/prompts/classification.txt` and `Code/prompts/extraction.txt`, few-shot examples in `Code/prompts/examples.json`
- Level 3 of the matching: `Code/pipeline/matching_prompt.py`
- Report generation: the prompt of every report is stored in `InputData/Reports.jsonl` and rebuilt by `Code/generation/`

## New reports

To run the pipeline on newly generated reports, change two lines in `Code/Config.py`: set `TEXT_DATEI` to `PAKET / "InputData/Generated/Reports.jsonl"` and `SPLIT_DATEI` to `None`.
