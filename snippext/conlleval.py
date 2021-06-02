"""
This script applies to IOB2 or IOBES tagging scheme.
If you are using a different scheme, please convert to IOB2 or IOBES.

IOB2:
- B = begin,
- I = inside but not the first,
- O = outside

e.g.
John   lives in New   York  City  .
B-PER  O     O  B-LOC I-LOC I-LOC O

IOBES:
- B = begin,
- E = end,
- S = singleton,
- I = inside but not the first or the last,
- O = outside

e.g.
John   lives in New   York  City  .
S-PER  O     O  B-LOC I-LOC E-LOC O

prefix: IOBES
chunk_type: PER, LOC, etc.
"""
from __future__ import division, print_function, unicode_literals

from collections import defaultdict


def split_tag(chunk_tag):
    """
    split chunk tag into IOBES prefix and chunk_type
    e.g.
    B-PER -> (B, PER)
    O -> (O, None)
    """
    return chunk_tag


def is_chunk_end(prev_tag, tag):
    """
    check if the previous chunk ended between the previous and current word
    e.g.
    (B, I) -> False
    (I, O)  -> True
    """
    return prev_tag == 'B' and tag == 'I'


def is_chunk_start(prev_tag, tag):
    """
    check if a new chunk started between the previous and current word
    """
    return prev_tag == 'O' and tag == 'B'


def calc_metrics(tp, p, t, percent=True):
    """
    compute overall precision, recall and FB1 (default values are 0.0)
    if percent is True, return 100 * original decimal value
    """
    precision = tp / p if p else 0
    recall = tp / t if t else 0
    fb1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    if percent:
        return 100 * precision, 100 * recall, 100 * fb1
    else:
        return precision, recall, fb1


def count_chunks(true_seqs, pred_seqs):
    """
    true_seqs: a list of true tags
    pred_seqs: a list of predicted tags

    return:
    correct_chunks: a dict (counter),
                    key = chunk types,
                    value = number of correctly identified chunks per type
    true_chunks:    a dict, number of true chunks per type
    pred_chunks:    a dict, number of identified chunks per type

    correct_counts, true_counts, pred_counts: similar to above, but for tags
    """
    # print(true_seqs)
    # print(pred_seqs)

    correct_chunks = defaultdict(int)
    true_chunks = defaultdict(int)
    pred_chunks = defaultdict(int)

    correct_counts = defaultdict(int)
    true_counts = defaultdict(int)
    pred_counts = defaultdict(int)

    prev_true_tag, prev_pred_tag = 'O', 'O'
    correct_chunk = None

    for true_tag, pred_tag in zip(true_seqs, pred_seqs):
        if true_tag == pred_tag:
            correct_counts[true_tag] += 1
        true_counts[true_tag] += 1
        pred_counts[pred_tag] += 1

        true_type = split_tag(true_tag)
        pred_type = split_tag(pred_tag)

        if correct_chunk is not None:
            true_end = is_chunk_end(prev_true_tag, true_tag)
            if true_end:
                true_chunks['BI'] += 1

            pred_end = is_chunk_end(prev_pred_tag, pred_tag)
            if pred_end:
                pred_chunks['BI'] += 1

            if pred_end and true_end:
                correct_chunks['BI'] += 1
                correct_chunk = None
            elif pred_end != true_end or true_type != pred_type:
                correct_chunk = None

        true_start = is_chunk_start(prev_true_tag, true_tag)
        pred_start = is_chunk_start(prev_pred_tag, pred_tag)

        if true_start and pred_start and true_type == pred_type:
            correct_chunk = true_type

        prev_true_tag, prev_pred_tag = true_tag, pred_tag
    if correct_chunk is not None:
        correct_chunks[correct_chunk] += 1

    print("Correct counts ", correct_counts, "True counts ", true_counts, "Pred counts", pred_counts)
    return (correct_chunks, true_chunks, pred_chunks,
            correct_counts, true_counts, pred_counts)


def get_result(correct_chunks, true_chunks, pred_chunks,
               correct_counts, true_counts, pred_counts, verbose=True):
    """
    if verbose, print overall performance, as well as preformance per chunk type;
    otherwise, simply return overall prec, rec, f1 scores
    """
    # sum counts
    sum_correct_chunks = sum(correct_chunks.values())
    sum_true_chunks = sum(true_chunks.values())
    sum_pred_chunks = sum(pred_chunks.values())

    sum_correct_counts = sum(correct_counts.values())
    sum_true_counts = sum(true_counts.values())

    nonO_correct_counts = sum(v for k, v in correct_counts.items() if k != 'O')
    nonO_true_counts = sum(v for k, v in true_counts.items() if k != 'O')

    chunk_types = sorted(list(set(list(true_chunks) + list(pred_chunks))))

    # compute overall precision, recall and FB1 (default values are 0.0)
    prec, rec, f1 = calc_metrics(correct_counts['B'] if correct_counts['B'] else 0, pred_counts['B'], true_counts['B'])
    res = (prec, rec, f1)
    if not verbose:
        return res

    # print overall performance, and performance per chunk type

    print("processed %i tokens;" % sum_true_counts, end='')
    # print("found: %i phrases; correct: %i.\n" % (sum_pred_chunks, sum_correct_chunks), end='')

    print("accuracy: %6.2f%%; (non-O)" % (100 * nonO_correct_counts / nonO_true_counts))
    print("accuracy: %6.2f%%; " % (100 * sum_correct_counts / sum_true_counts), end='')
    print("precision: %6.2f%%; recall: %6.2f%%; FB1: %6.2f" % (prec, rec, f1))

    # for each chunk type, compute precision, recall and FB1 (default values are 0.0)
    # for t in chunk_types:
    #     prec, rec, f1 = calc_metrics(sum_correct_counts[t], pred_counts[t], true_counts[t])
    #     print("%17s: " % t, end='')
    #     print("precision: %6.2f%%; recall: %6.2f%%; FB1: %6.2f" %
    #           (prec, rec, f1), end='')
    #     print("  %d" % pred_chunks[t])
    #
    return res


def evaluate(true_seqs, pred_seqs, verbose=True):
    (correct_chunks, true_chunks, pred_chunks,
     correct_counts, true_counts, pred_counts) = count_chunks(true_seqs, pred_seqs)
    result = get_result(correct_chunks, true_chunks, pred_chunks,
                        correct_counts, true_counts, pred_counts, verbose=verbose)
    return result


def evaluate_conll_file(fileIterator):
    true_seqs, pred_seqs = [], []

    for line in fileIterator:
        cols = line.strip().split()
        if not cols:
            true_seqs.append('O')
            pred_seqs.append('O')
        elif len(cols) < 3:
            raise IOError("conlleval: too few columns in line %s\n" % line)
        else:
            true_seqs.append(cols[-2])  # coloana 1
            pred_seqs.append(cols[-1])  # coloana 2
    return evaluate(true_seqs, pred_seqs)