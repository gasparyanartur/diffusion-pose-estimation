from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from scripts.investigation.information_suite.summarize_pilot import interval


def classify_cross_validated(features, labels, groups, reduce=False):
    repo = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo/'tmp/h_control_information_suite/deps'))
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    features, labels, groups = np.asarray(features), np.asarray(labels), np.asarray(groups)
    unique = list(dict.fromkeys(groups))
    folds = {family:index % 8 for index, family in enumerate(unique)}
    assigned = np.array([folds[group] for group in groups])
    scores = np.zeros(len(labels))
    for fold in range(8):
        test = assigned == fold
        if not test.any():
            continue
        scaler = StandardScaler().fit(features[~test])
        train, heldout = scaler.transform(features[~test]), scaler.transform(features[test])
        if reduce:
            pca = PCA(n_components=16, random_state=42, svd_solver='randomized').fit(train)
            train, heldout = pca.transform(train), pca.transform(heldout)
        model = LogisticRegression(C=1, max_iter=2000, random_state=42).fit(train, labels[~test])
        scores[test] = model.decision_function(heldout)
    prediction = scores > 0
    pair_scores, family_accuracy = [], []
    for family in unique:
        selected = groups == family
        local_scores, local_labels = scores[selected], labels[selected]
        assert len(local_labels) == 2
        difference = local_scores[local_labels == 1][0]-local_scores[local_labels == 0][0]
        pair_scores.append(.5 if difference == 0 else float(difference > 0))
        family_accuracy.append(float((prediction[selected] == local_labels).mean()))
    return dict(accuracy=float((prediction == labels).mean()), correct=int((prediction == labels).sum()),
                images=len(labels), family_accuracy=interval(family_accuracy), paired_order=interval(pair_scores),
                folds=8, split='whole physical family', PCA16=reduce, C=1)


def pilot_cues(root, manifest):
    data = dict(rgb=[], outside_arm=[], raw2D=[], labels=[], groups=[], strata=[])
    for family in manifest['families']:
        for member in (0, 1):
            entry = family['conditions'][f'main_{member}']
            image = Image.open(entry['path']).convert('RGB')
            data['rgb'].append(np.asarray(image.crop((100, 35, 300, 145)).resize((96, 48)), dtype=np.float32).reshape(-1)/255)
            outside = np.concatenate([np.asarray(image.crop((100, 0, 300, 50)).resize((32, 8))).reshape(-1),
                                      np.asarray(image.crop((100, 150, 300, 256)).resize((32, 16))).reshape(-1)])
            data['outside_arm'].append(outside.astype(np.float32)/255)
            coordinates = np.asarray(entry['keypoints']).reshape(-1)
            data['raw2D'].append(np.r_[coordinates, np.asarray(entry['boxes']).reshape(-1)])
            data['labels'].append(member)
            data['groups'].append(family['family_id'])
            data['strata'].append(family['stratum'])
    result = {}
    for source in ('rgb', 'outside_arm', 'raw2D'):
        result[source] = classify_cross_validated(data[source], data['labels'], data['groups'], reduce=source != 'raw2D')
    if manifest['study'] == 'C':
        for stratum in ('surface', 'occlusion'):
            selected = np.asarray(data['strata']) == stratum
            result['rgb_'+stratum] = classify_cross_validated(np.asarray(data['rgb'])[selected],
                np.asarray(data['labels'])[selected], np.asarray(data['groups'])[selected], reduce=True)
        raw = result['raw2D']
        result['raw2D_leakage_invalidates'] = raw['paired_order']['mean'] > .55 and raw['paired_order']['ci95'][0] > .5
    else:
        result['raw2D']['interpretation'] = 'Wrist coordinates differ by design in B; this is NOT a B image-only control.'
    result['configuration'] = 'Fixed8-fold family CV, train-fold standardization, PCA16 for RGB only, logistic C=1.'
    result['outside_arm_interpretation'] = 'Audit only: above-chance scores may indicate render/pose/shadow shortcuts away from intended arm.'
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--cues-only', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((args.root/'pilot_manifest.json').read_text())
    assert manifest['complete'] and len(manifest['families']) == 32
    cue_path = args.root/'cue_audit.json'
    if not cue_path.exists():
        cue_path.write_text(json.dumps(pilot_cues(args.root, manifest), indent=2)+'\n')
    cues = json.loads(cue_path.read_text())
    if args.cues_only:
        print(json.dumps(cues, indent=2))
        return
    models = {}
    assert (args.root/'native'/'DONE.json').exists(), 'Native evaluation is not complete'
    for arm in ('control', 'h'):
        rows = [json.loads(line) for line in (args.root/'native'/f'{arm}.jsonl').read_text().splitlines()]
        selected = [row['people'][0] for row in rows if row['condition'].startswith('clear_')]
        models[arm] = dict(images=len(selected), detected=sum(r['detected'] for r in selected),
                           correct=sum(r['wrist_correct'] for r in selected))
        models[arm]['passed'] = models[arm]['correct']/len(selected) >= .85
    positive = cues['rgb']['accuracy'] >= .85
    if manifest['study'] == 'C':
        positive &= all(cues['rgb_'+stratum]['accuracy'] >= .85 for stratum in ('surface', 'occlusion'))
    leaked = cues.get('raw2D_leakage_invalidates', False)
    admitted = positive and not leaked and all(model['passed'] for model in models.values())
    output = dict(study=manifest['study'], admitted=admitted, RGB_positive_control_passed=bool(positive),
                  raw2D_leakage_invalidates=leaked, models=models,
                  visual_audit_required=True, note='Admission is numerical only until visual/physical/source parity audits also pass.')
    (args.root/'admission.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
