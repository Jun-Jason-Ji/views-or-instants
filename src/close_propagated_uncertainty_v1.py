"""Test, verify and seal the completed failed candidate without changing it."""
import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'experiments/propagated_uncertainty_v1_2026-09-14'
ALE = ROOT/'experiments/huh7_01_budget_smoke_v1_2026-09-14'


def stamp(): return datetime.now(timezone.utc).isoformat()
def load(p): return json.loads(p.read_text(encoding='utf-8'))
def digest(p):
    with p.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
def save(p, value):
    with p.open('x', encoding='utf-8') as f: json.dump(value, f, indent=2, allow_nan=False)


def tests():
    code = """import json,time,unittest,sys
start=time.perf_counter()
suite=unittest.defaultTestLoader.discover('tests')
result=unittest.TextTestRunner(verbosity=2).run(suite)
print(json.dumps(dict(tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),unexpected_successes=len(result.unexpectedSuccesses),passed=result.wasSuccessful(),wall_s=time.perf_counter()-start)))
sys.exit(0 if result.wasSuccessful() else 1)
"""
    python = ROOT/'.venvs/learned_match/Scripts/python.exe'
    result = subprocess.run([str(python), '-c', code], cwd=ROOT, capture_output=True, text=True, encoding='utf-8', errors='replace')
    with (OUT/'full_tests.log').open('x', encoding='utf-8') as f: f.write(result.stderr+'\n'+result.stdout)
    assert result.returncode == 0, result.stderr[-5000:]
    record = json.loads(result.stdout.strip().splitlines()[-1]); assert record['passed']
    save(OUT/'full_tests.json', dict(**record, utc=stamp(), python=str(python), log_sha256=digest(OUT/'full_tests.log')))
    print(json.dumps(record), flush=True)


def close():
    tick = perf_counter(); verified = []; memo = {}
    prior = [ROOT/'experiments/weighted_rigid_cpu_v1_2026-09-13/results_seal.json',
        ROOT/'experiments/fr3_weighted_confirmation_v1_2026-09-13/results_seal.json',
        ROOT/'experiments/q_error_structure_v1_2026-09-14/results_seal.json',
        ROOT/'experiments/relative_pose_diagnosis_v1_2026-09-14/results_seal.json',
        ROOT/'experiments/huh7_scoring_core_v1_2026-09-14/artifact_seal.json',
        ROOT/'experiments/huh7_scoring_core_v2_2026-09-14/artifact_seal.json']
    current = [OUT/n for n in ['freeze.json', 'prediction_seal.json', 'evaluation_seal.json', 'shared_frame_audit_seal.json', 'independent_review_seal.json']]
    current += [ALE/n for n in ['freeze.json', 'prediction_seal.json', 'evaluation_seal.json', 'artifact_seal.json']]
    for seal in prior+current:
        data = load(seal)
        hashes = data['hashes'] if 'hashes' in data else {r['path']: r['sha256'] for r in data['files']}
        for rel, sha in hashes.items():
            path = (ROOT/rel).resolve(); assert path.is_relative_to(ROOT), rel
            if path not in memo: memo[path] = digest(path)
            assert memo[path] == sha, rel
        verified.append(dict(path=seal.relative_to(ROOT).as_posix(), file_count=len(hashes), sha256=digest(seal)))
    test = load(OUT/'full_tests.json'); assert test['passed'] and test['failures'] == test['errors'] == 0
    assert digest(OUT/'full_tests.log') == test['log_sha256']
    review = load(OUT/'independent_review.json')
    assert review['status'] == 'pass_current_results_and_failed_gate_independently_reproduced' and review['counts']['prediction_rows'] == 600
    stable = load(OUT/'independent_stable_gp_review.json'); assert stable['status'] == 'pass' and stable['all_q_choices_equal']
    ale_review = load(ALE/'root_independent_review.json'); assert ale_review['verdict'] == 'pass_no_blocking_issue'
    assert digest(ALE/'root_independent_review.py') == ale_review['script_sha256']
    assert load(OUT/'decision.json')['status'] == 'development_gate_failed_do_not_promote'
    assert load(OUT/'validation.json')['technical_pass'] and load(ALE/'validation.json')['technical_pass']
    report = ROOT/'reports/PROPAGATED_UNCERTAINTY_V1_2026-09-14.md'
    paths = [report, ROOT/'src/report_propagated_uncertainty_v1.py', OUT/'uncertainty_controls.png', OUT/'uncertainty_controls.pdf']
    save(OUT/'report_seal.json', dict(utc=stamp(), hashes={p.relative_to(ROOT).as_posix(): digest(p) for p in paths}))
    save(OUT/'completion.json', dict(utc=stamp(), status='complete_failed_development_candidate_no_promotion',
        tests=test, independent_review_sha256=digest(OUT/'independent_review.json'), stable_qr_review_sha256=digest(OUT/'independent_stable_gp_review.json'),
        stable_qr_max_path_delta_m=stable['max_selected_path_delta_m'], stable_qr_max_all_q_nll_delta=stable['max_all_q_nll_delta'], prior_and_current_seals_verified=verified,
        distinct_files_verified=len(memo), technical_rows=600, scientific_gate='failed',
        inference_runtime_s=load(OUT/'runtime.json')['wall_s'], new_gpu_calls=0, model_training=False,
        new_tum_image_decodes=0, new_huh7_01_image_decodes=24, new_huh7_02_payload_decodes=0,
        ale=dict(status='budget_reader_classical_detector_common_time_scoring_smoke_complete_not_strong_baseline',
            report='reports/HUH7_01_BUDGET_SMOKE_V1_2026-09-14.md', budgets=[16, 8], reference_times_scored_per_budget=8,
            root_independent_review_sha256=digest(ALE/'root_independent_review.json')),
        qa=dict(png_viewed=True, labels_legible=True, pdf_same_figure=True),
        shared_frame_cache='660adjacentedgepairs;median sharedIDs657/68.57%ofsmallerrawsupport;394397sharedoccurrences;allsharedXYZexact;datareadiness not covarianceidentification.',
        nonblocking_scope_note='No fallback occurred. Config fallback wording says rawposes; prediction positions field actually retains iid smoothed positions. Public rawposition metrics always use old geometry. Clarify schema in a new version before any future deployment; current zero-fallback scores unaffected.',
        next='Fixed CPU point-influence model with shared-frame cross-edge terms, compare matched marginals/without cross terms, synthetic PSD/perturbation checks before freezing any next evaluation. Retain strongLG/depth/Huber controls. ALE strongmodel/identity/officialCTC and02 freeze remain pending. No large-scale training.',
        closure_wall_s=perf_counter()-tick))
    artifacts = [p for p in OUT.rglob('*') if p.is_file() and '__pycache__' not in p.parts]+paths+[Path(__file__), ROOT/'reports/HUH7_01_BUDGET_SMOKE_V1_2026-09-14.md', ALE/'root_independent_review.py', ALE/'root_independent_review.json']
    save(OUT/'results_seal.json', dict(utc=stamp(), hashes={p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(set(artifacts))}))
    print(json.dumps(dict(tests=test['tests_run'], verified_unique_files=len(memo), result_files=len(set(artifacts)), results_seal_sha256=digest(OUT/'results_seal.json'))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['tests', 'close']); globals()[parser.parse_args().action]()
