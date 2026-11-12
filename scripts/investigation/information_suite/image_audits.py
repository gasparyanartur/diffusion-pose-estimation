from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from scripts.investigation.information_suite.readouts import fit_reader, paired_scores
from scripts.investigation.information_suite.summarize_pilot import interval


def load_inputs(root):
    manifest = json.loads((root/'pilot_manifest.json').read_text())
    raw, outside, rows = [], [], []
    for family in manifest['families']:
        for condition in ('main_0', 'main_1', 'nuisance_0', 'nuisance_1'):
            entry = family['conditions'][condition]
            image = Image.open(entry['path']).convert('RGB')
            pixels = np.r_[np.asarray(image.crop((100, 0, 300, 50)).resize((32, 8))).reshape(-1),
                           np.asarray(image.crop((100, 150, 300, 256)).resize((32, 16))).reshape(-1)]
            outside.append(pixels.astype(np.float32)/255)
            raw.append(np.r_[np.asarray(entry['keypoints']).reshape(-1), np.asarray(entry['boxes']).reshape(-1)])
            rows.append(dict(family_id=family['family_id'], label=int(condition[-1]), condition=condition))
    return np.array(raw), np.array(outside), rows


def main():
    parser = argparse.ArgumentParser()
    for name in ('train', 'val', 'test', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    data = [load_inputs(root) for root in (args.train, args.val, args.test)]
    repo = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo/'tmp/h_control_information_suite/deps'))
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    train_main = np.array([row['condition'].startswith('main_') for row in data[0][2]])
    scaler = StandardScaler().fit(data[0][1][train_main])
    pca = PCA(n_components=16, random_state=42, svd_solver='randomized').fit(scaler.transform(data[0][1][train_main]))
    features = dict(raw2D=[split[0] for split in data],
                    outside_arm=[pca.transform(scaler.transform(split[1])) for split in data])
    result = dict(literal_shared2D_pair_order=.5, scope='Audits use all physical families, independent of native box correspondence.')
    for source, values in features.items():
        (normalize, model, strength), validation = fit_reader(values[0], values[1], data[0][2], data[1][2], paired_scores)
        scores = model.decision_function(normalize.transform(values[2]))
        _, pairs = paired_scores(scores, data[2][2])
        result[source] = dict(C=strength, validation_pair_order=validation, test_pair_order=interval(pairs),
                              test_image_accuracy=float(((scores > 0) == np.array([row['label'] for row in data[2][2]])).mean()))
    raw = result['raw2D']['test_pair_order']
    result['raw2D_leakage_invalidates'] = raw['mean'] > .55 and raw['ci95'][0] > .5
    result['outside_arm_warning'] = result['outside_arm']['test_pair_order']['ci95'][0] > .5
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
