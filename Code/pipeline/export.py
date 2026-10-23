import csv
import io
from . import benchmark, results, matching_runner
from .experiment import load, matching_context

COLUMNS = ('run', 'method', 'task', 'shots', 'baseline', 'training_percent', 'training_n', 'rung', 'n_expected', 'n_scored', 'n_pending', 'complete', 'accuracy', 'invalid', 'api_errors', 'api_failed_reports_pending', 'mean_latency_sec', 'timing_n_requests', 'timing_n_api_errors', 'timing_n_length_stops', 'n_wall_time_sec', 'sum_wall_time_sec', 'mean_wall_time_sec', 'median_wall_time_sec', 'p95_wall_time_sec', 'n_ollama_total_sec', 'sum_ollama_total_sec', 'mean_ollama_total_sec', 'median_ollama_total_sec', 'p95_ollama_total_sec', 'n_ollama_load_sec', 'sum_ollama_load_sec', 'mean_ollama_load_sec', 'median_ollama_load_sec', 'p95_ollama_load_sec', 'n_ollama_prompt_eval_sec', 'sum_ollama_prompt_eval_sec', 'mean_ollama_prompt_eval_sec', 'median_ollama_prompt_eval_sec', 'p95_ollama_prompt_eval_sec', 'n_ollama_eval_sec', 'sum_ollama_eval_sec', 'mean_ollama_eval_sec', 'median_ollama_eval_sec', 'p95_ollama_eval_sec', 'n_ollama_inference_sec', 'sum_ollama_inference_sec', 'mean_ollama_inference_sec', 'median_ollama_inference_sec', 'p95_ollama_inference_sec', 'n_ollama_total_without_load_sec', 'sum_ollama_total_without_load_sec', 'mean_ollama_total_without_load_sec', 'median_ollama_total_without_load_sec', 'p95_ollama_total_without_load_sec', 'n_wall_minus_ollama_sec', 'sum_wall_minus_ollama_sec', 'mean_wall_minus_ollama_sec', 'median_wall_minus_ollama_sec', 'p95_wall_minus_ollama_sec', 'n_ollama_other_sec', 'sum_ollama_other_sec', 'mean_ollama_other_sec', 'median_ollama_other_sec', 'p95_ollama_other_sec', 'n_prompt_tokens', 'sum_prompt_tokens', 'n_completion_tokens', 'sum_completion_tokens', 'n_prompt_cached_tokens', 'sum_prompt_cached_tokens', 'n_generation_tokens_per_sec', 'generation_tokens_per_sec', 'accuracy_Activity', 'accuracy_Machine', 'accuracy_Error time and Error date', 'accuracy_WorkerID', 'macro_f1', 'macro_f1_all_classes', 'validation_n', 'validation_percent', 'pass3_method', 'total', 'assigned', 'correct', 'wrong_assigned', 'unassigned', 'coverage', 'precision_assigned', 'component_wall_sec', 'local_python_sec', 'ollama_server_sec', 'ollama_load_sec', 'compute_sec', 'compute_without_load_sec', 'client_transport_residual_sec', 'model_management_sec', 'llm_tokens')


def result_rows(c, folder):
    number = int(folder.name.removeprefix('run_'))
    metric_path = folder / 'benchmark/metrics.json'
    rows = [dict(row, run=number) for row in load(metric_path)['results']] if metric_path.exists() else []
    if (folder / 'benchmark/config.json').exists():
        evaluation = benchmark.read_rows(folder / 'benchmark/records/eval.jsonl')
        context = matching_context(c, folder)
        jobs = matching_runner.build_jobs(context, load(folder / 'benchmark/config.json'))
        for row in results.matching_records(folder / 'matching', jobs):
            row['run'] = number
            row['baseline'] = None if row['method'].startswith('llm_') else row['method'].split('__train_')[0]
            count = len(evaluation) if str(row['rung']) == 'all' else sum(str(r['difficulty_level']) == str(row['rung']) for r in evaluation)
            row['n_expected'] = count
            row['n_scored'] = row.get('total') if row['complete'] else 0
            row['n_pending'] = count - (row['n_scored'] or 0)
            rows.append(row)
    return sorted(rows, key=lambda r: (r['task'], r['method'], r.get('pass3_method') or '',
                                      0 if str(r['rung']) == 'all' else int(r['rung'])))


def csv_bytes(rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=COLUMNS, extrasaction='ignore', lineterminator='\n')
    writer.writeheader()
    for row in rows:
        values = {}
        for key in COLUMNS:
            value = row.get(key)
            if isinstance(value, bool):
                value = 'true' if value else 'false'
            elif isinstance(value, float):
                value = format(value, '.15g')
            values[key] = value
        writer.writerow(values)
    return stream.getvalue().encode('utf-8')


def export_run(c, folder):
    path = c.ERGEBNIS_ORDNER / (folder.name + '.csv')
    benchmark.atomic(path, csv_bytes(result_rows(c, folder)))
    return path
