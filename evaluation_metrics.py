'''
Implementation based on:

Information Retrieval WS 17/18, Lecture 2: Ranking and Evaluation lesson,
by Prof. Dr. Hannah Bast at the University of Freiburg, Germany.
Available on: https://www.youtube.com/watch?v=bCVPnnWqY8s&t=4629s (accessed on 24th october, 2018).

and

Chen, Ching-Wei & Lamere, Paul & Schedl, Markus & Zamani, Hamed. (2018).
"Recsys challenge 2018: automatic music playlist continuation".
527-528. 10.1145/3240323.3240342.
'''
import numpy as np

# -----------------------------------------------------
# DCG function
# -----------------------------------------------------
def dcg_sum(vec, rel):
    return np.sum(rel[vec]/np.log2(vec+1))

def compute_DCG(rel):
    DCG = 0.0
    if n_correct_predictions == 0:
        return DCG

    if correct_indeces[0] == 0:
        if n_correct_predictions == 1:
            DCG = rel[0]
        else:
            DCG = rel[0] + dcg_sum(correct_indeces[1:], rel)
    else:
        DCG = dcg_sum(correct_indeces, rel)
    return DCG

# -----------------------------------------------------
# iDCG function
# -----------------------------------------------------
def compute_iDCG(rel):
    iDCG = 0.0
    if n_correct_predictions == 0:
        return iDCG

    rel_ordered = np.sort(rel)[::-1]

    if n_correct_predictions == 1:
        iDCG = rel_ordered[0]
    else:
        iDCG = rel_ordered[0] + \
               np.sum( rel_ordered[1:n_correct_predictions] / np.log2(np.arange(2, n_correct_predictions+1)) )
    return iDCG

# -----------------------------------------------------
# nDCG function
# -----------------------------------------------------
def compute_nDCG(size, rel=[]):
    if len(rel) == 0: # Use default weights (ones)
        rel = np.ones(size)

    DCG = compute_DCG(rel)
    iDCG = compute_iDCG(rel)

    try:
        nDCG = DCG/iDCG*100
    except:
        nDCG = 0.0

    return nDCG

# -----------------------------------------------------
# Precision @ K function
# -----------------------------------------------------
def compute_precision(ref, dst, k):
    precision_at_k = 0.0
    if k < 1 or k > len(dst):
        return precision_at_k
    else:
        dst = dst[:k]
    precision_at_k = len(np.where(np.in1d(dst, ref))[0])/k*100
    return precision_at_k

# -----------------------------------------------------
# Average Precision function
# -----------------------------------------------------
def compute_average_precision(ref, dst):
    average_precision = 0.0
    size = len(correct_indeces)
    if size == 0:
        return average_precision
    precision_at_k = np.array([compute_precision(ref, dst, k) for k in np.arange(1, size+1)])
    average_precision = np.sum(precision_at_k) / size
    return average_precision

# -----------------------------------------------------
# R-Precision function
# -----------------------------------------------------
def compute_r_precision(dst, REF, DST, weight = 0.25):
    intersection_specific_group_size = len(correct_indeces)
    intersection_supper_group_size = 0.0

    if len(REF) != 0 and len(DST) != 0:
        intersection_supper_group_size = len(np.where(np.in1d(DST, REF))[0])

    r_precision = (intersection_specific_group_size + (weight * intersection_supper_group_size)) / len(dst) * 100
    return r_precision

# -----------------------------------------------------
# Main function
# -----------------------------------------------------
def main(ref, dst, rel = [], k = 1, REF = [], DST = [], weight = 0.25):
    global correct_indeces
    global n_correct_predictions

    correct_indeces = np.where(np.in1d(dst, ref))[0]
    n_correct_predictions = len(correct_indeces)

    ndcg = compute_nDCG(len(dst), rel)
    precision_at_k = compute_precision(ref, dst, k)
    average_precision = compute_average_precision(ref, dst)
    r_precision = compute_r_precision(dst, REF, DST, weight)

    print('NDCG: {0}'.format(ndcg))
    print('Precision@{0}: {1}'.format(k, precision_at_k))
    print('Average Precision: {0}'.format(average_precision))
    print('R-Precision: {0}'.format(r_precision))
    print('')


if __name__ == '__main__':
    main(np.array([1, 2, 3, 4, 5]), np.array([1, 4, 6, 2, 7]), rel = np.array([2, 1, 0, 2, 0]))
    main(np.array([1, 2, 3, 4, 5]), np.array([0, 2, 6, 7, 3]), rel = np.array([0, 1, 1, 0, 1]))
    main(np.array([1, 2]), np.array([0, 2, 6, 7, 3]), rel = np.array([0, 1, 1, 0, 1]))

    # It's works without pass relevance values, default weights will be used
    main(np.array([1, 2]), np.array([0, 2, 6, 7, 3]))

    # It's works with non-numbers too
    main(np.array(['yellow', 'gray', 'red', 'black', 'blue']), np.array(['red', 'green', 'blue', 'orange', 'pink']))

    # Correct precision based on Youtube lesson:
    # P@1=100%; P@2=50%; P@3=33%; P@4=50%; P@5=60%
    main(np.array(['a', 'b', 'c', 'd']), np.array(['a', 'e', 'f', 'd', 'a', 'g']), k = 1)
    main(np.array(['a', 'b', 'c', 'd']), np.array(['b', 'e', 'f', 'd', 'a', 'g']), k = 2)
    main(np.array(['a', 'b', 'c', 'd']), np.array(['b', 'e', 'f', 'd', 'a', 'g']), k = 3)
    main(np.array(['a', 'b', 'c', 'd']), np.array(['b', 'e', 'f', 'd', 'a', 'g']), k = 4)
    main(np.array(['a', 'b', 'c', 'd']), np.array(['b', 'e', 'f', 'd', 'a', 'g']), k = 5)

    # Approach used to evaluate submissions in RecSys Challenge 2018
    # More details at: https://recsys-challenge.spotify.com/
    main(np.array(['aa', 'ab', 'ac', 'ba', 'bb']), np.array(['ac', 'ad', 'ba', 'bc', 'cc']), \
         REF = np.array(['A', 'A', 'A', 'B', 'B']), DST = np.array(['A', 'A', 'B', 'B', 'C']), weight = 0.0)
    main(np.array(['aa', 'ab', 'ac', 'ba', 'bb']), np.array(['ac', 'ad', 'ba', 'bc', 'cc']), \
         REF = np.array(['A', 'A', 'A', 'B', 'B']), DST = np.array(['A', 'A', 'B', 'B', 'C']), weight = 0.25)
    main(np.array(['aa', 'ab', 'ac', 'ba', 'bb']), np.array(['ac', 'ad', 'ba', 'bc', 'cc']), \
         REF = np.array(['A', 'A', 'A', 'B', 'B']), DST = np.array(['A', 'A', 'B', 'B', 'C']), weight = 0.5)
    main(np.array(['aa', 'ab', 'ac', 'ba', 'bb']), np.array(['ac', 'ad', 'ba', 'bc', 'cc']), \
         REF = np.array(['A', 'A', 'A', 'B', 'B']), DST = np.array(['A', 'A', 'B', 'B', 'C']), weight = 1.0)