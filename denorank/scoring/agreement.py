from scipy.stats import spearmanr


def rank_agreement(influence, score):
    """Spearman rank correlation between TracIn influence and a candidate score.

    For the denoising residual (lower = better) a strong *negative* rho means
    agreement with TracIn.
    """
    rho, p = spearmanr(influence, score)
    return float(rho), float(p)
