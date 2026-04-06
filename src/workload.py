import csv
import os
from collections import deque

import constants
import definitions as defs

JOBS_WORKLOAD = []
JOB_QUEUE = deque()


def _to_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _workload_path():
    return os.path.join(constants.root, 'input', constants.workload)


def _load_csv_rows(path):
    with open(path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        if reader.fieldnames:
            return list(reader), True
        csvfile.seek(0)
        raw_reader = csv.reader(csvfile, delimiter=',')
        return [row for row in raw_reader if row], False


def parse_synthetic_trace(rows, is_dict_rows):
    """Parse synthetic traces in the original schema."""
    jobs = []
    if is_dict_rows:
        for idx, row in enumerate(rows):
            arrival = _to_int(row.get('arrival_time', row.get('a_t', 0)))
            job_id = _to_int(row.get('job_id', row.get('id', idx)))
            job_type = _to_int(row.get('job_type', row.get('type', 1)), 1)
            cpu = _to_int(row.get('cpu', 1), 1)
            mem = _to_int(row.get('mem', 1), 1)
            ex = _to_int(row.get('executors', row.get('ex', 1)), 1)
            duration = _to_int(row.get('duration', 1), 1)
            jobs.append(defs.JOB(arrival, job_id, job_type, cpu, mem, ex, duration))
    else:
        for row in rows:
            if len(row) < 7:
                continue
            jobs.append(
                defs.JOB(
                    _to_int(row[0]),
                    _to_int(row[1]),
                    _to_int(row[2], 1),
                    _to_int(row[3], 1),
                    _to_int(row[4], 1),
                    _to_int(row[5], 1),
                    _to_int(row[6], 1)
                )
            )
    return jobs


def parse_google_trace(rows, is_dict_rows):
    """Parse Google-like traces and map them into JOB objects."""
    jobs = []
    if not is_dict_rows:
        return parse_synthetic_trace(rows, is_dict_rows)

    for idx, row in enumerate(rows):
        arrival = _to_int(row.get('submit_time', row.get('arrival_time', 0)))
        job_id = _to_int(row.get('job_id', row.get('id', idx)))
        # Infer a lightweight type proxy from priority if available.
        priority = _to_int(row.get('priority', 1), 1)
        job_type = 1 + (priority % 3)
        cpu = max(1, _to_int(row.get('cpu_request', row.get('cpu', 1)), 1))
        mem = max(1, _to_int(row.get('memory_request', row.get('mem', 1)), 1))
        ex = max(1, _to_int(row.get('task_count', row.get('executors', 1)), 1))
        duration = max(1, _to_int(row.get('runtime', row.get('duration', 1)), 1))
        jobs.append(defs.JOB(arrival, job_id, job_type, cpu, mem, ex, duration))
    return jobs


def parse_alibaba_trace(rows, is_dict_rows):
    """Parse Alibaba-like traces and map them into JOB objects."""
    jobs = []
    if not is_dict_rows:
        return parse_synthetic_trace(rows, is_dict_rows)

    for idx, row in enumerate(rows):
        start_time = _to_int(row.get('start_time', row.get('arrival_time', 0)))
        end_time = _to_int(row.get('end_time', start_time + 1), start_time + 1)
        duration = max(1, end_time - start_time)
        job_id = _to_int(row.get('job_id', row.get('id', idx)))
        # Map instance/task type hints into one of the original 3 types.
        task_type = _to_int(row.get('task_type', 1), 1)
        job_type = 1 + (task_type % 3)
        cpu = max(1, _to_int(row.get('cpu', row.get('plan_cpu', 1)), 1))
        mem = max(1, _to_int(row.get('mem', row.get('plan_mem', 1)), 1))
        ex = max(1, _to_int(row.get('instance_num', row.get('executors', 1)), 1))
        jobs.append(defs.JOB(start_time, job_id, job_type, cpu, mem, ex, duration))
    return jobs


def get_job_batch(start_idx, batch_size):
    """Return up to batch_size pending jobs from the loaded workload."""
    if batch_size <= 0:
        return []
    end_idx = min(len(JOBS_WORKLOAD), start_idx + batch_size)
    return JOBS_WORKLOAD[start_idx:end_idx]


def read_workload():
    path = _workload_path()
    rows, is_dict_rows = _load_csv_rows(path)

    trace_type = (constants.trace_type or 'synthetic').strip().lower()
    if trace_type == 'google':
        jobs = parse_google_trace(rows, is_dict_rows)
    elif trace_type == 'alibaba':
        jobs = parse_alibaba_trace(rows, is_dict_rows)
    else:
        jobs = parse_synthetic_trace(rows, is_dict_rows)

    jobs.sort(key=lambda job: (job.arrival_time, job.id))

    global JOBS_WORKLOAD
    global JOB_QUEUE
    JOBS_WORKLOAD = jobs
    JOB_QUEUE = deque(jobs)
