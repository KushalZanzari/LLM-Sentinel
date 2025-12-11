import yaml

def load_thresholds():
    with open('configs/thresholds.yaml', 'r') as f:
        return yaml.safe_load(f)


def compute_quality_score(rel: float, comp: float, factual_avg: float) -> float:
    """
    Weighted quality score between 0 and 1:
      40% relevance, 30% completeness, 30% factuality
    """
    score = 0.4 * rel + 0.3 * comp + 0.3 * factual_avg
    # clamp 0..1
    return max(0.0, min(1.0, float(score)))


def make_verdict(rel, comp, factual):
    """
    Determines PASS / WARN / FAIL based on thresholds.
    factual is the avg factuality score (float)
    """
    th = load_thresholds()

    rel_ok = rel >= th['relevance_min']
    comp_ok = comp >= th['completeness_min']
    fact_ok = factual >= th['factuality_min']

    # FAIL if factuality is too low
    if not fact_ok:
        return 'FAIL'

    # WARN if relevance or completeness low
    if not rel_ok or not comp_ok:
        return 'WARN'

    # otherwise everything is good
    return 'PASS'


def build_report(rel, comp, factual_report, latency, token_usage):
    """
    Final structured JSON report. Adds quality_score.
    """
    fact_avg = factual_report['avg_score']
    fact_hall = factual_report['hallucinated_claims']

    quality_score = compute_quality_score(rel, comp, fact_avg)

    verdict = make_verdict(rel, comp, fact_avg)

    return {
        'scores': {
            'relevance': rel,
            'completeness': comp,
            'factuality': {
                'avg_score': fact_avg,
                'claims': factual_report['claims'],
                'claim_scores': factual_report['claim_scores'],
                'hallucinated_claims': fact_hall,
            },
            'quality_score': quality_score
        },
        'latency_seconds': latency,
        'token_usage': token_usage,
        'verdict': verdict,
    }
