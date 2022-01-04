from utils import *
from goatools.semantic import resnik_sim

DiGraph  = NewType('nx DiGraph', nx.DiGraph)
CLabelDict = NewType('LabelDict w/ Confidences or Counts', Dict[int, List[Tuple[str, int]]])

def compute_auprc(precision: List[float], recall: List[float])-> float:
    auprc, p_rec = 0,0
    for i in range(len(recall)):
        auprc += precision[i] * (recall[i] - p_rec)
        p_rec  = recall[i]
    return auprc

def compute_f1max(conf_labels: CLabelDict, real_labels: Dict[str, List[str]],
        fast: bool=False, n_intervals: int=100)-> Union[ndarray, ndarray, float, float]:
    ps,rs = list(), list()

    for i in np.linspace(0,1, n_intervals + ((n_intervals+1)%2)):
        pc, rc, prs, rcs = 0, 0, 0, 0
        for idx, labels in conf_labels.items():
            pred = set([go_id for (go_id,conf) in labels if conf >= i])
            if len(pred) > 0: pc += 1
            real = set(real_labels[idx])
            if len(real) > 0: rc += 1

            overlap = float(len(real & pred))
            prs += overlap/float(len(pred)) if len(pred) > 0 else 0
            rcs += overlap/float(len(real)) if len(real) > 0 else 0
        try: ps.append(prs/pc); rs.append(rcs/rc)
        except ZeroDivisionError: rs.append(np.nan); ps.append(np.nan)

    return np.nanmax([(2*p*r)/(p+r) if p+r!=0 else 0 for (p,r) in zip(ps,rs)])

# Remember to set the correct namespace!!! Must be one of {'BP', 'MF', 'CC'}

def rsnk(predicted, true, go_dag, term_counts):
    try: return resnik_sim(predicted, true, go_dag, term_counts)
    except KeyError: return -1.0 # guard against missing/obsolete terms

def sem_similarity(pred_labels, true_labels, go_dag, term_counts):
    sims1 = [np.max([rsnk(p,t,go_dag,term_counts) for t in true_labels]) for p in pred_labels]
    sims2 = [np.max([rsnk(t,p,go_dag,term_counts) for p in pred_labels]) for t in true_labels]
    return (np.sum(sims1) + np.sum(sims2)) / float(len(sims1) + len(sims2))

def resnik_score(test_idxs, tree_pred, tree_true, go_dag, term_counts, n_intervals: int = 20):
    """Scores cross validation by counting the number of test nodes that were accurately labeled
    after their removal from the true labelling. """
    
    ## EXAMPLE ##
    # go_dag = GODag("../go-basic.obo.dat")
    # term_counts = TermCounts(go_dag, read_gaf("/path_to_your_gaf_file", namespace="BP"))

    conf_intervals, sims = [i/n_intervals for i in range(n_intervals)], list()
    for c in conf_intervals:
        sim = sim_counter = 0
        for t_idx in test_idxs:
            pred_labels = {g for (g, c1) in tree_pred.get(t_idx, [('N/A', 0)]) if c1>=c}-{'N/A'}
            true_labels = tree_true.get(t_idx, [])
            if len(true_labels) == 0 or len(pred_labels) == 0: continue
            else:
                sim_counter += 1
                sim += sem_similarity(pred_labels, true_labels, go_dag, term_counts)
        try:
            if sim_counter < 10: raise ZeroDivisionError
            sims.append(sim / float(sim_counter))
        except ZeroDivisionError: sims.append(np.nan)
    return np.nanmax(sims)

# newer resnik version 7/6/21
def compute_metrics(predictions: CLabelDict, target_GO_labels: Dict[int, str],
                    go_dag = None, term_counts = None, n_labels: int=None)-> ndarray:
    n_correct, n_predicted, tree_pred, tree_true, ridxs = 0, 0, dict(), dict(), set()

    for test_idx, predicted_labels in predictions.items():
        ridxs.add(test_idx)
        (winner, _) = predicted_labels[0]
        conf_labels = predicted_labels[:n_labels]
        real_labels = target_GO_labels.get(test_idx, None)

        if real_labels is not None:
            n_predicted += 1
            if winner in real_labels: n_correct += 1 # used for accuracy
            tree_pred[test_idx], tree_true[test_idx] = conf_labels, real_labels

    if n_predicted == 0: print('\t+++ EMPTY +++'); return np.array([-1.0,-1.0], dtype=float)
    
    # NOTE: Uncomment for accuracy and f1max - can handle all three GO aspects simultaneously
    A = (100 * n_correct) / n_predicted
    F = compute_f1max(tree_pred, tree_true) # remember that P and R are arrays of size <intervals>
    # print(np.asarray([float(A), float(F)], dtype=float))
    return np.asarray([float(A), float(F)], dtype=float)
    
    
    # NOTE: Uncomment for resnik - can only handle one GO aspect at a time
    # R = resnik_score(ridxs, tree_pred, tree_true, go_dag, term_counts) # remember that P and R are arrays of size <intervals>
    #print(np.asarray([float(R)], dtype=float))
    # return np.asarray([float(R)], dtype=float)

