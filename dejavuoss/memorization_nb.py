import argparse
import os, json, csv
import numpy as np
import operator
from collections import Counter
from pathlib import Path
from collections import defaultdict
from pathlib import Path


'''
Command to run the script

python memorization_nb.py --input_path "outputs_knn_success" \
					         --output_path "outputs_knn_success" \
  						 --imagenet_label_path /datasets01/imagenet_full_size/061417/labels.txt \
						 --label_topk 3  \
 						 --smooth 1e-10 
'''

def most_conf_frac(scores, frac): 
    """get topk NN predictions on the most confident fraction
    of attacked examples. Run after compute_topk_preds
    Input:
        most_conf_frac: scalar [0,1], most confident frac of examps
    Return: 
        frac_idxs: indices of the most confident examples
        preds: topk predictions of these examples 
    """
    n_most_conf = int(frac * len(scores))
        
    #get most confident subset of indices
    most_conf_idxs = np.argsort(scores)[::-1][:n_most_conf]

    #get predictions 
    most_conf_preds = scores[most_conf_idxs]

    return most_conf_idxs, most_conf_preds

def get_xclass_crop_labels(input_path: Path,
                           folder2label_lst: dict):
    total_crop_labels_counter = Counter()
    for folder2label in folder2label_lst:
        class_id, class_name = folder2label
        freq_input_path = os.path.join(input_path, 'frequencies', f'{class_id}_{class_name}_freq.json')
        freq_input = open(freq_input_path)
        freqs = Counter({item[0]: item[1][0] for item in json.loads(freq_input.read()).items()})
        total_crop_labels_counter +=  freqs

    return total_crop_labels_counter

def predict_mem_score(input_path: Path,
                      folder2label_lst: dict,
                      xclass_crop_labels_counter: dict,
                      label_topk: int,
                      smooth: float = 1e-10,
                      laplace=False,
                      alpha = 1):

    freq_maps = defaultdict(defaultdict)
    pred_class_tp_counts = Counter()

    total_image_counter = 0.0
    tp = 0.0

    prob_vals_all = []
    prob_keys_all = []
    targets_all = []

    print('laplace: ', laplace)

    # Create directory to save predictions
    predictions_dir = f'predictions_top_{label_topk}_laplace_{str(laplace)}'
    Path(os.path.join(input_path, predictions_dir)).mkdir(parents=True, exist_ok=True)

    for folder2label in folder2label_lst:
        class_id, class_name = folder2label
        class_key_true_label = f'{class_id}_{class_name}'

        freq_input_path = os.path.join(input_path, 'frequencies', f'{class_key_true_label}_freq.json')
        freq_input = open(freq_input_path)
        freq_map = json.loads(freq_input.read())
        freq_maps[class_key_true_label] = freq_map

    smooth_double = [smooth] * 2
    # iterate over all possible predicted labels
    for folder2label in folder2label_lst:
        class_id, class_name = folder2label
        class_key_true_label = f'{class_id}_{class_name}'
        labels_input_path = os.path.join(input_path, 'labels', f'{class_key_true_label}_labels.json')
        print('labels_input_path: ', labels_input_path)
        if not os.path.exists(labels_input_path):
            print('There are no examples avaialable under the path: ', labels_input_path)
            continue
        label_input = open(labels_input_path)
        label_data = json.loads(label_input.read())

        example2prob = defaultdict(defaultdict)
        example2maxprob = defaultdict(float)

        for example in label_data:
            image_path = example['image_path']
            tags = example['tags']
            scores = example['scores']
            tags = np.array(tags)[np.argsort(scores)[-label_topk:]]
            print('top tags: ', tags)

            total_image_counter += 1.0
            for freq_key, freq_value in freq_maps.items():
                joint_prob = 1.0
                for tag in tags:
                    tag = tag.strip()
                    print('freq_value.get(tag): ', freq_value.get(tag))
                    if not laplace:
                        total_cls = xclass_crop_labels_counter.get(tag, smooth_double[0])
                    else:    
                        total_cls = xclass_crop_labels_counter.get(tag, 0.0) + label_topk * alpha
                    
                    if not laplace:
                        tag_pred_class_freq = freq_value.get(tag, smooth_double)[0]
                    else:
                        tag_pred_class_freq = freq_value.get(tag, [0.0, 0.0])[0] + alpha

                    prob = tag_pred_class_freq / total_cls
                    joint_prob *= prob
                example2prob[image_path][freq_key] = 0.0 if len(tags) == 0 else joint_prob

            example2prob_vals, example2prob_keys = list(example2prob[image_path].values()), \
                                                   list(example2prob[image_path].keys())

            example2maxprob[image_path] = max(zip(example2prob_vals, example2prob_keys))
            score_pred, class_pred = example2maxprob[image_path]
            example2maxprob[image_path] += (example2prob[image_path][class_key_true_label], class_key_true_label)

            prob_vals_all.append(score_pred)
            prob_keys_all.append(class_pred)
            targets_all.append(class_key_true_label)

            if class_pred == class_key_true_label:
                tp += 1.0
                pred_class_tp_counts[class_key_true_label] += 1

        predictions_input_path = os.path.join(input_path, predictions_dir, f'{class_key_true_label}_predictions.json')
        with open(predictions_input_path, 'w') as fp:
            json.dump(example2maxprob, fp)

    pred_class_tp_counts_sorted = [sorted(pred_class_tp_counts.items(), key=operator. itemgetter(1), reverse = True)]

    pred_class_tp_counts_path = os.path.join(input_path, f'pred_class_tp_counts_top{label_topk}_laplace_{str(laplace)}.json')
    with open(pred_class_tp_counts_path, 'w') as fp:
            json.dump(pred_class_tp_counts_sorted, fp)

    print('Accuracy: ', tp / total_image_counter)

    return np.array(prob_keys_all), np.array(prob_vals_all), np.array(targets_all)


def compute_accuracy(accs_targets, predictions, accs_indices_subset):
    print('predictions[accs_indices_subset]: ', len(predictions[accs_indices_subset]))
    print('accs_targets[accs_indices_subset]: ', len(accs_targets[accs_indices_subset]))
    return (predictions[accs_indices_subset] == accs_targets[accs_indices_subset]).mean()

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Memorization measurement with Naive Bayes", add_help=True)
    parser.add_argument("--input_path", type=Path, required=True, help="path to input directors")
    parser.add_argument("--output_path", type=Path, required=True, help="path to output directors")
    parser.add_argument("--imagenet_label_path", required=True, type=str, help="path to imagenet label mapping")
    parser.add_argument("--label_topk", required=False, type=int, default=1, help="the number of top labels that are used to compute nb probabilities")
    parser.add_argument("--smooth", required=False, type=float, default=1e-10, help="Count smoothing value used to estimate the probabilities")
    parser.add_argument("--laplace", action='store_true', help="Use laplace smoothing or not")
    parser.add_argument("--alpha", required=False, type=int, default=1, help="Alpha for laplace smoothing")

    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    imagenet_label_path = args.imagenet_label_path
    label_topk = args.label_topk
    smooth = args.smooth
    alpha = args.alpha
    laplace = args.laplace

    with open(imagenet_label_path) as folder2label:
        folder2label_reader = csv.reader(folder2label, delimiter=',')
        folder2label_lst = [line for line in folder2label_reader]

    xclass_label2cnt = get_xclass_crop_labels(input_path, folder2label_lst)

    freq_path = os.path.join(output_path, 'class_counts.json')
    with open(freq_path, 'w') as fp:
        json.dump(xclass_label2cnt, fp)

    predictions, scores, targets = predict_mem_score(input_path,
                                                     folder2label_lst,
                                                     xclass_label2cnt,
                                                     label_topk, 
                                                     smooth=smooth,
                                                     alpha=alpha,
                                                     laplace=laplace)

    print('scores: ', scores)
    score_indices_subset, most_conf_scores = most_conf_frac(scores, 1)
    print('score_indices_subset: ', score_indices_subset)
    print('Crop model Acc on Test: ', compute_accuracy(targets, predictions, score_indices_subset))

    score_indices_subset, most_conf_scores = most_conf_frac(scores, 0.05)
    print('Crop model Acc on Test 5%: ', compute_accuracy(targets, predictions, score_indices_subset))

    score_indices_subset, most_conf_scores = most_conf_frac(scores, 0.2)
    print('Crop model Acc on Test 20%: ', compute_accuracy(targets, predictions, score_indices_subset))

    #example2prob_sorted = [sorted(example2prob.items(), key=operator. itemgetter(1), reverse = True)]
    #for example, probs in example2prob.items():
    #    print('example: ', example, probs)

    '''
    print('Saving sorted naive bayes probability scores for each example')
    probs_path = os.path.join(output_path, f'probs_top{label_topk}.json')
    with open(probs_path, 'w') as fp:
        json.dump(example2prob_sorted, fp)
    '''