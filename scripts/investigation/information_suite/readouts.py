from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from scripts.investigation.information_suite.summarize_pilot import interval


def load_cache(root, arm):
    prefix = root/'features'/arm
    arrays = {key:value for key,value in np.load(prefix.with_suffix('.npz')).items()}
    rows = json.loads(prefix.with_name(prefix.name+'_rows.json').read_text())
    return arrays, rows


def paired_scores(scores, rows, condition='main'):
    families = {}
    for score, row in zip(scores, rows):
        if row['condition'].startswith(condition+'_'):
            families.setdefault(row['family_id'], {})[row['label']] = float(score)
    ids, values = [], []
    for family, members in families.items():
        if len(members) != 2:
            continue
        difference = members[1]-members[0]
        ids.append(family)
        values.append(.5 if difference == 0 else float(difference > 0))
    return ids, np.asarray(values)


def classified_pairs(scores, rows, condition='main'):
    families = {}
    for score, row in zip(scores, rows):
        if row['condition'].startswith(condition+'_'):
            families.setdefault(row['family_id'], []).append(int(score > 0) == row['label'])
    ids = [family for family, members in families.items() if len(members) == 2]
    return ids, np.array([float(all(families[family])) for family in ids])


def shuffle_other_families(values, rows, seed):
    rng = np.random.default_rng(seed)
    families = np.array([row['family_id'] for row in rows])
    for _ in range(10000):
        permutation = rng.permutation(len(rows))
        if np.all(families != families[permutation]):
            return values[permutation]
    raise RuntimeError('Cannot construct independent-family row permutation')


def feature_variants(arrays, rows, seed):
    variants = dict(arrays)
    if 'h_written_1' in arrays:
        wrist = arrays['G2']
        variants['G2_plus_elbow'] = np.c_[wrist, arrays['G2_elbow']]
        variants['G2_plus_h'] = np.c_[wrist, arrays['h_written_1']]
        shuffled = shuffle_other_families(arrays['h_written_1'], rows, seed)
        variants['G2_plus_shuffled_h'] = np.c_[wrist, shuffled]
        variants['G2_three_joints'] = np.c_[arrays['G2_shoulder'], arrays['G2_elbow'], wrist]
    return variants


def fit_reader(train, validation, train_rows, validation_rows, endpoint):
    repo = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo/'tmp/h_control_information_suite/deps'))
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    train_mask = np.array([row['condition'].startswith('main_') for row in train_rows])
    labels = np.array([row['label'] for row in train_rows])[train_mask]
    scaler = StandardScaler().fit(train[train_mask])
    transformed = scaler.transform(train[train_mask])
    heldout = scaler.transform(validation)
    selected, best = None, -np.inf
    for strength in (.01, .1, 1., 10., 100.):
        model = LogisticRegression(C=strength, max_iter=2000, random_state=42).fit(transformed, labels)
        scores = model.decision_function(heldout)
        _, pairs = endpoint(scores, validation_rows)
        accuracy = float(pairs.mean())
        if accuracy > best:
            selected, best = (scaler, model, strength), accuracy
    return selected, best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--val', type=Path, required=True)
    parser.add_argument('--test', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--study', choices=['B', 'C'], required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    endpoint = classified_pairs if args.study == 'B' else paired_scores
    arms = ['control', 'h'] + (['initialization'] if args.study == 'C' else [])
    caches = {arm:[load_cache(root, arm) for root in (args.train, args.val, args.test)] for arm in arms}
    retained = []
    for split_index in range(3):
        families = None
        for arm in arms:
            rows = caches[arm][split_index][1]
            accepted = {row['family_id'] for row in rows}
            accepted -= {row['family_id'] for row in rows if not row['matched']}
            families = accepted if families is None else families & accepted
        retained.append(families)
        alignment = [[(row['family_id'], row['condition']) for row in caches[arm][split_index][1]
                      if row['family_id'] in families] for arm in arms]
        assert all(rows == alignment[0] for rows in alignment)
    result = dict(study=args.study, models={}, retention={})
    predictions = {}
    for arm in arms:
        variants, rows_by_split = [], []
        for split_index, (arrays, rows) in enumerate(caches[arm]):
            mask = np.array([row['family_id'] in retained[split_index] for row in rows])
            kept = [row for row, keep in zip(rows, mask) if keep]
            values = {key:value[mask] for key,value in arrays.items()}
            variants.append(feature_variants(values, kept, 4200+split_index) if args.study == 'C' else values)
            rows_by_split.append(kept)
        train, validation, test = variants
        model_results = {}
        for name in train:
            (scaler, model, strength), validation_score = fit_reader(train[name], validation[name], rows_by_split[0], rows_by_split[1], endpoint)
            scores = model.decision_function(scaler.transform(test[name]))
            ids, values = endpoint(scores, rows_by_split[2])
            _, nuisance = endpoint(scores, rows_by_split[2], 'nuisance')
            labels = np.array([row['label'] for row in rows_by_split[2]])
            main_mask = np.array([row['condition'].startswith('main_') for row in rows_by_split[2]])
            strata = {}
            for stratum in ('surface', 'occlusion'):
                family_strata = {row['family_id']:row['stratum'] for row in rows_by_split[2]}
                selected = [value for family,value in zip(ids,values) if family_strata[family] == stratum]
                if selected:
                    strata[stratum] = interval(selected)
            model_results[name] = dict(C=strength, dimensions=train[name].shape[1], validation_pair_accuracy=validation_score,
                endpoint='both_members_classified' if args.study == 'B' else 'within_pair_depth_order',
                main_pair_accuracy=interval(values), nuisance_pair_accuracy=interval(nuisance), paired_nuisance_drop=interval(values-nuisance),
                main_image_accuracy=float(((scores > 0) == labels)[main_mask].mean()), strata=strata)
            predictions[f'{arm}_{name}'] = scores
            np.savez_compressed(args.output.parent/f'{arm}_{name}_fit.npz', mean=scaler.mean_, scale=scaler.scale_,
                                coefficient=model.coef_, intercept=model.intercept_, C=strength)
        result['models'][arm] = model_results
        result['retention'][arm] = {split:dict(original_families=len({r['family_id'] for r in caches[arm][index][1]}),
                retained_families=len(retained[index])) for index,split in enumerate(('train', 'val', 'test'))}
    comparisons = [('h_G2', 'control_G2')]
    if args.study == 'C':
        comparisons += [(f'{arm}_{source}', f'initialization_{source}')
                        for arm in ('control', 'h') for source in ('B', 'E', 'G1', 'G2')]
        comparisons += [('h_G2_plus_h', 'h_G2_plus_elbow'),
                        ('h_G2_plus_h', 'h_G2_plus_shuffled_h')]
    result['paired_contrasts'] = {}
    for first, second in comparisons:
        _, first_values = endpoint(predictions[first], rows_by_split[2])
        _, second_values = endpoint(predictions[second], rows_by_split[2])
        result['paired_contrasts'][first+' minus '+second] = interval(first_values-second_values)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    np.savez_compressed(args.output.with_suffix('.scores.npz'), **predictions)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
