import numpy as np

def ranking_precision_score(y_true, y_score, k=10, skip=None, use_min=False):
    """Precision at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    Returns
    -------
    precision @k : float
    """

    init1 = y_true.sum()
    if skip is not None:
        y_true = np.array([x for i,x in enumerate(y_true) if i not in skip])
        y_score = np.array([x for i,x in enumerate(y_score) if i not in skip])

    init2 = y_true.sum()
    unique_y = np.unique(y_true)

    if len(unique_y) > 2:
        raise ValueError("Only supported for two relevance levels.")

    try:
        pos_label = unique_y[1]
    except:
        return 0.
    n_pos = np.sum(y_true == pos_label)

    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    init3 = y_true.sum()
    n_relevant = np.sum(y_true == pos_label)

    # Divide by min(n_pos, k) such that the best achievable score is always 1.0.
    #return float(n_relevant) / min(float(k), n_pos)
    if skip is not None:
        '''
        # len_skip: 50 (10,) (6992,) [ 0.  1.]
        # n_rel: 7
        # n_pos: 21
        # K: 10
        # init1: 31.0
        # init2: 21.0
        #init3: 7.0
        # pos_Label: 1.0
        55 (10,) (6987,) [ 0.  1.] 0 1 10 56.0 1.0 0.0 1.0
        30 (10,) (7012,) [ 0.  1.] 0 1 10 31.0 1.0 0.0 1.0
        '''
        #print len(skip), y_true.shape, y_score.shape, unique_y, n_relevant, n_pos, k, init1, init2, init3, pos_label
        abc = 1
    if use_min and False:
        return float(n_relevant) / min(n_pos, k)
    else:
        return float(n_relevant) / k

def ranking_recall_score(y_true, y_score, k=10, skip=None, use_min=False):
    """Precision at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    Returns
    -------
    precision @k : float
    """
    if skip is not None:
        y_true = np.array([x for i,x in enumerate(y_true) if i not in skip])
        y_score = np.array([x for i,x in enumerate(y_score) if i not in skip])
    unique_y = np.unique(y_true)

    if len(unique_y) > 2:
        raise ValueError("Only supported for two relevance levels.")

    try:
        pos_label = unique_y[1]
    except:
        return 0.
    n_pos = np.sum(y_true == pos_label)

    order = np.argsort(y_score)[::-1]
    total = np.sum(y_true) 
    y_true = np.take(y_true, order[:k])
    n_relevant = np.sum(y_true == pos_label)

    # Divide by min(n_pos, k) such that the best achievable score is always 1.0.
    if use_min and False:
        return float(n_relevant) / min(float(n_pos), float(k))
    else:
        return float(n_relevant) / float(n_pos)

def pak(y_true, y_score, k=10, skip=None, use_min=True):
    paks = []
    if skip is not None:
        for t, p, s in zip(y_true, y_score, skip):
            paks.append(ranking_precision_score(t, p, k, s, use_min=use_min))
    else:
        for t, p in zip(y_true, y_score):
            paks.append(ranking_precision_score(t, p, k,  use_min=use_min))
    return np.mean(paks)

def rak(y_true, y_score, k=10, skip=None, use_min=False):
    raks = []
    if skip is not None:
        for t, p, s in zip(y_true, y_score, skip):
            raks.append(ranking_recall_score(t, p, k, s,  use_min=use_min))
    else:
        for t, p in zip(y_true, y_score):
            raks.append(ranking_recall_score(t, p, k, None,  use_min=use_min))
    return np.mean(raks)
