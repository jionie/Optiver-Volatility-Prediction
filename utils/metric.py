from scipy.stats import pearsonr, spearmanr


def pearson_correlation(pred, label):
    correlation, p_value = pearsonr(pred, label)
    return correlation, p_value


def spearman_correlation(pred, label):
    correlation, p_value = spearmanr(pred, label)
    return correlation, p_value
