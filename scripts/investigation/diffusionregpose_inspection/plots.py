from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np
from PIL import Image
import torch

from scripts.investigation.diffusionregpose_inspection.run import load_trace
from scripts.investigation.diffusionregpose_inspection.trace import query_lineage
from src.data.skeleton import CROWDPOSE_EDGE_LIST, CROWDPOSE_JOINT_NAMES

JOINT_NAMES = list(CROWDPOSE_JOINT_NAMES)
JOINT_COLORS = plt.get_cmap('tab20')(np.linspace(0, 1, 14))


def sample_for(cache, image_id):
    return next(sample for sample in cache['samples'] if sample['image_id'] == image_id)


def image_axis(axis, sample):
    axis.imshow(Image.open(sample['img_path']).convert('RGB'))
    height, width = sample['img_hw']
    axis.set(xlim=(0, width), ylim=(height, 0))
    axis.set_axis_off()


def draw_pose(axis, points, visibility=None, *, color=None, alpha=1, labels=False):
    points = np.asarray(points)
    valid = np.isfinite(points).all(-1)
    if visibility is not None:
        valid &= np.asarray(visibility) > 0
    for first, second in CROWDPOSE_EDGE_LIST:
        if valid[first] and valid[second]:
            axis.plot(*points[[first, second]].T, color=color or 'white', lw=1.2, alpha=alpha)
    axis.scatter(*points[valid].T, c=JOINT_COLORS[valid] if color is None else color,
                 s=19, edgecolors='black', linewidths=.25, alpha=alpha)
    if visibility is not None:
        occluded = valid & (np.asarray(visibility) == 1)
        axis.scatter(*points[occluded].T, s=35, facecolors='none', edgecolors='white', alpha=alpha)
    if labels:
        for index in np.flatnonzero(valid):
            axis.annotate(str(index), points[index], color='white', fontsize=7, xytext=(3, 3), textcoords='offset points')


def draw_box(axis, box, *, color='cyan', label=None):
    center_x, center_y, width, height = np.asarray(box)
    axis.add_patch(Rectangle((center_x-width/2, center_y-height/2), width, height,
                             fill=False, edgecolor=color, linewidth=1.4))
    axis.scatter([center_x], [center_y], marker='+', c=color, s=45)
    if label:
        axis.text(center_x-width/2, center_y-height/2, label, color='black', fontsize=8,
                  bbox={'facecolor': color, 'alpha': .8, 'pad': 1})


def gallery_figure(cache, steps, *, threshold=.25):
    figure, axes = plt.subplots(4, 3, figsize=(14, 15), squeeze=False)
    scores = {row['image_id']: row['AP'] for row in cache['rows'] if row['steps'] == steps}
    for cohort_index, cohort in enumerate(('best', 'worst')):
        for rank, image_id in enumerate(cache['rankings'][steps][cohort]):
            sample = sample_for(cache, image_id)
            ground_axis, predicted_axis = axes[2*cohort_index, rank], axes[2*cohort_index+1, rank]
            for axis in (ground_axis, predicted_axis):
                image_axis(axis, sample)
            for points, visible in zip(sample['keypoints'], sample['visibility']):
                draw_pose(ground_axis, points, visible)
            result = cache['sweep'][(image_id, steps)]['result']
            shown = (result['labels'] == 1) & (result['scores'] >= threshold)
            for points, score in zip(result['keypoints'][shown], result['scores'][shown]):
                points = points.reshape(14, 3)
                draw_pose(predicted_axis, points[:, :2])
                predicted_axis.text(*points[13, :2], f'{score:.2f}', color='yellow', fontsize=8)
            ground_axis.set_title(f'{cohort.title()} {rank+1} · image {image_id} · AP {scores[image_id]:.3f}\nGround truth')
            predicted_axis.set_title(f'Prediction · {int(shown.sum())} people shown (score ≥ {threshold})')
    figure.suptitle(f'{steps} diffusion step(s) · ranked within the same 32 test images\n'
                     'All native detections enter AP; the display threshold affects only this gallery.', fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, .95))
    return figure


def geometry_figure(cache, image_id, steps, call_index, person, *, zoom=True):
    call = load_trace(cache, image_id, steps)[call_index]
    sample = sample_for(cache, image_id)
    scale = torch.tensor(sample['img_hw'][::-1])
    figure, axes = plt.subplots(len(call['layers']), 3, figsize=(13, 3.6*len(call['layers'])), squeeze=False)
    final_box = call['layers'][-1]['prediction']['pred_boxes'][person] * scale.repeat(2)
    center_x, center_y, box_width, box_height = final_box.tolist()
    zoom_x = (max(0, center_x-box_width*.8), min(float(scale[0]), center_x+box_width*.8))
    zoom_y = (min(float(scale[1]), center_y+box_height*.8), max(0, center_y-box_height*.8))
    for index, layer in enumerate(call['layers']):
        for axis in axes[index]:
            image_axis(axis, sample)
        query = query_lineage(call, index, person)
        reference = layer['references'][query]
        draw_box(axes[index, 0], reference * scale.repeat(2), label=f'input query {query}')
        prediction = layer['prediction']
        prediction_query = int(call['selected_boxes'][person]) if index < call['num_box_layers'] else person
        confidence = prediction['pred_logits'][prediction_query, 1].sigmoid()
        draw_box(axes[index, 1], prediction['pred_boxes'][prediction_query] * scale.repeat(2),
                 label=f'person confidence {confidence:.3f}')
        if index >= call['num_box_layers']:
            start = query + 1
            references = layer['references'][start:start+14]
            draw_pose(axes[index, 0], references[:, :2] * scale, labels=True)
            reference_pixels = references[:, :2] * scale
            outside = ((reference_pixels < 0) | (reference_pixels > scale)).any(-1)
            if outside.any():
                lower = torch.minimum(reference_pixels.amin(0), torch.zeros(2))
                upper = torch.maximum(reference_pixels.amax(0), scale)
                margin = (upper-lower)*.05
                axes[index, 0].set_facecolor('#e8e8e8')
                axes[index, 0].set(xlim=(float(lower[0]-margin[0]), float(upper[0]+margin[0])),
                                   ylim=(float(upper[1]+margin[1]), float(lower[1]-margin[1])))
                axes[index, 0].text(.02, .02, f'{int(outside.sum())}/14 references outside image\nAxes expanded to show actual coordinates',
                                     transform=axes[index, 0].transAxes, fontsize=8,
                                     bbox={'facecolor': 'white', 'alpha': .9})
            keypoints = prediction['pred_keypoints'][person, :28].reshape(14, 2) * scale
            draw_pose(axes[index, 2], keypoints, labels=True)
            # The decoder carries per-joint XYWH references, not only point centers.
            for ref, color in zip(references, JOINT_COLORS):
                x, y, width, height = (ref * scale.repeat(2)).tolist()
                axes[index, 0].add_patch(Rectangle((x-width/2, y-height/2), width, height,
                                                 fill=False, edgecolor=color, alpha=.35, linewidth=.6))
        else:
            axes[index, 2].text(.5, .5, 'Box-only layer\nNo joint prediction', transform=axes[index, 2].transAxes,
                                ha='center', color='white', bbox={'facecolor': 'black', 'alpha': .8})
        for column, title in enumerate(('Incoming references (XY + extents)', 'Predicted box', 'Predicted joints')):
            axes[index, column].set_title(f'Layer {index+1} · {title}', fontsize=10)
        if zoom:
            for axis in axes[index, 1:]:
                axis.set(xlim=zoom_x, ylim=zoom_y)
    figure.suptitle(f'Image {image_id} · S={steps} · call {call_index+1}/{steps} · t={call["time"]} · group {person}\n'
                     'Group labels are local to this call; identical labels across calls do not establish identity.', fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, .94))
    return figure


def proposal_figure(cache, image_id, steps, person):
    calls = load_trace(cache, image_id, steps)
    sample = sample_for(cache, image_id)
    scale = torch.tensor(sample['img_hw'][::-1])
    diffusion_scale = 6.
    figure, axes = plt.subplots(2, steps, figsize=(5*steps, 8), squeeze=False)
    for index, call in enumerate(calls):
        for axis in axes[:, index]:
            image_axis(axis, sample)
        points = (call['state'][person].reshape(14, 2)/diffusion_scale + 1)/2 * scale
        draw_pose(axes[0, index], points, labels=True)
        off_image = ((points < 0) | (points > scale)).any(-1).sum()
        axes[0, index].set_title(f'Call {index+1} · t={call["time"]}\nNoisy state slot {person} · {off_image}/14 outside image')
        points = call['layers'][-1]['prediction']['pred_keypoints'][person, :28].reshape(14, 2)*scale
        draw_pose(axes[1, index], points, labels=True)
        axes[1, index].set_title('Clean joint estimate (before DDIM update)')
    figure.suptitle(f'Image {image_id} · S={steps} · diffusion state versus clean prediction\n'
                     'Each slot is local to the call; decoder references are shown separately below.')
    figure.tight_layout(rect=(0, 0, 1, .93))
    return figure


def mask_figure(cache, image_id, steps, call_index, person):
    call = load_trace(cache, image_id, steps)[call_index]
    figure, axes = plt.subplots(len(call['layers']), 2, figsize=(12, 4*len(call['layers'])), squeeze=False)
    for index, layer in enumerate(call['layers']):
        count = len(layer['references'])
        mask = torch.zeros(count, count, dtype=torch.bool) if layer['mask'] is None else layer['mask']
        axes[index, 0].imshow(mask, cmap='gray_r', vmin=0, vmax=1, interpolation='nearest')
        axes[index, 0].set_title(f'Layer {index+1}: {count} × {count} · '
                                  + ('no mask (all allowed)' if layer['mask'] is None else 'actual mask, identical across heads'))
        if index >= call['num_box_layers']:
            group = call['num_joints'] + 1
            groups = sorted({person, (person+1) % len(call['selected_boxes'])})
            indices = torch.cat([torch.arange(item*group, (item+1)*group) for item in groups])
            crop = mask[indices][:, indices]
            labels = [f'{g}:box' if j == 0 else f'{g}:{j-1}' for g in groups for j in range(group)]
        else:
            indices = call['selected_boxes'][:min(15, len(call['selected_boxes']))]
            crop = mask[indices][:, indices]
            labels = [str(int(query)) for query in indices]
        axes[index, 1].imshow(crop, cmap='gray_r', vmin=0, vmax=1, interpolation='nearest')
        axes[index, 1].set_xticks(range(len(labels)), labels, rotation=90, fontsize=6)
        axes[index, 1].set_yticks(range(len(labels)), labels, fontsize=6)
        axes[index, 1].set_title('Query subset · group:joint labels; box tokens can read other boxes')
        for axis in axes[index]:
            axis.set_xlabel('Key query'); axis.set_ylabel('Reading query')
    figure.suptitle(f'Image {image_id} · S={steps} · call {call_index+1} · self-attention masks\n'
                     'White = allowed (False); black = blocked (True). Masks do not show learned attention strength.')
    figure.tight_layout(rect=(0, 0, 1, .94))
    return figure


SAMPLE_COLORS = ['#ff851b', '#2ca02c', '#9467bd', '#d62728']


def person_attention_context(sample, prediction, person, reference, locations):
    image_wh = np.array(sample['img_hw'][::-1], dtype=np.float32)
    points = prediction['pred_keypoints'][person, :28].reshape(14, 2).numpy()*image_wh
    ground_truth = np.asarray(sample['keypoints'])
    labeled = np.asarray(sample['visibility']) > 0
    matched_person = None
    person_coordinates = points
    if len(ground_truth) and labeled.any():
        distances = np.linalg.norm((ground_truth-points[None])/image_wh, axis=-1)
        errors = (distances*labeled).sum(-1)/np.maximum(labeled.sum(-1), 1)
        errors[labeled.sum(-1) == 0] = np.inf
        matched_person = int(errors.argmin())
        box = np.asarray(sample['gt_boxes_pixel'][matched_person])
        corners = np.array([box[:2], box[:2]+box[2:]])
        person_coordinates = np.vstack([points, ground_truth[matched_person][labeled[matched_person]], corners])
    lower, upper = person_coordinates.min(0), person_coordinates.max(0)
    padding = np.maximum((upper-lower)*.1, 6)
    person_lower = np.maximum(lower-padding, 0)
    person_upper = np.minimum(upper+padding, image_wh)
    # Keep recognizable face/shoulder context in the attention view as well.
    detail_coordinates = np.vstack([reference[None], locations.reshape(-1, 2), points[[0, 1, 12, 13]]])
    lower, upper = detail_coordinates.min(0), detail_coordinates.max(0)
    span = np.maximum(upper-lower, (person_upper-person_lower)*np.array([.6, .4]))
    center = (upper+lower)/2
    return {'ground_truth_person': matched_person, 'prediction_points': points,
            'person_limits': (person_lower, person_upper),
            'detail_limits': (center-span*.65, center+span*.65)}


def person_attention_figure(cache, image_id, steps, call_index, person, *, joint=4, head=0,
                            layer_index=2, feature_level=None):
    call = load_trace(cache, image_id, steps)[call_index]
    sample = sample_for(cache, image_id)
    layer = call['layers'][layer_index]
    if joint is not None and layer_index < call['num_box_layers']:
        return attention_figure(cache, image_id, steps, call_index, person,
                                joint=joint, head=head, layer_index=layer_index)
    query = query_lineage(call, layer_index, person, joint)
    weights = layer['weights'][query, head].numpy()
    level = int(weights.sum(-1).argmax()) if feature_level is None else feature_level
    if not 0 <= level < len(weights):
        raise ValueError('feature level is out of range')
    image_wh = np.array(sample['img_hw'][::-1], dtype=np.float32)
    reference = layer['references'][query, :2].numpy()*image_wh
    locations = layer['locations'][query, head, level].numpy()*image_wh
    context = person_attention_context(sample, call['layers'][-1]['prediction'], person, reference, locations)
    figure = plt.figure(figsize=(12, 8))
    grid = figure.add_gridspec(2, 2, width_ratios=[1, 1.35], height_ratios=[3, 1.15],
                               left=.045, right=.97, top=.84, bottom=.16, hspace=.48, wspace=.32)
    person_axis = figure.add_subplot(grid[:, 0])
    detail_axis = figure.add_subplot(grid[0, 1])
    strength_axis = figure.add_subplot(grid[1, 1])
    image = np.asarray(Image.open(sample['img_path']).convert('RGB'))
    image_axis(person_axis, sample)
    person_axis.set_gid('attention-person')
    draw_pose(person_axis, context['prediction_points'], color='#00dcff')
    if joint is not None:
        target = context['prediction_points'][joint]
        person_axis.scatter(*target, s=190, facecolors='none', edgecolors='yellow', linewidths=2.5, zorder=5)
    lower, upper = context['person_limits']
    person_axis.set(xlim=(lower[0], upper[0]), ylim=(upper[1], lower[1]))
    person_axis.set_title('The person and predicted pose\n'
                          + ('Yellow ring: prediction · green X: annotation' if joint is not None else 'Selected person box query'), fontsize=11)
    lower, upper = context['detail_limits']
    draw_attention_samples(detail_axis, image, reference, locations, lower, upper, weights=weights[level])
    annotated_joint = None
    matched = context['ground_truth_person']
    if joint is not None and matched is not None and sample['visibility'][matched][joint] > 0:
        annotated_joint = np.asarray(sample['keypoints'][matched][joint])
        for axis in (person_axis, detail_axis):
            axis.scatter(*annotated_joint, marker='X', s=125, c='#39ff14',
                         edgecolors='black', linewidths=.8, zorder=8)
    outside = bool(((reference < 0) | (reference > image_wh)).any())
    token = 'Person box' if joint is None else JOINT_NAMES[joint].replace('_', ' ').capitalize()
    detail_axis.set_title(f'{token}: reference and sampling offsets\n'
                          f'Decoder layer {layer_index+1} · head {head} · feature level {level+1}'
                          + ('\nReference lies outside the image' if outside else ''), fontsize=12)
    draw_sample_strengths(strength_axis, weights[level])
    strength_axis.set_xlabel('Attention strength (% of all 16 samples in this head)')
    legend = [Line2D([], [], marker='*', markerfacecolor='white', markeredgecolor='black',
                     markersize=13, linestyle='none', label='Reference point'),
              Line2D([], [], color='#0088ff', linewidth=2, marker='>', label='Predicted offset')]
    if annotated_joint is not None:
        legend.append(Line2D([], [], marker='X', markerfacecolor='#39ff14', markeredgecolor='black',
                             markersize=10, linestyle='none', label='Annotated joint'))
    legend.extend(Line2D([], [], marker='o', color=color, markeredgecolor='black', markersize=8,
                         linestyle='none', label=f'Sample {index+1}') for index, color in enumerate(SAMPLE_COLORS))
    figure.legend(handles=legend, loc='lower center', bbox_to_anchor=(.5, .025),
                  ncol=4, frameon=False, fontsize=10)
    figure.suptitle(f'Attention on a person · image {image_id}\n'
                     f'{token} · diffusion call {call_index+1} of {steps} · '
                     + ('strongest feature level shown' if feature_level is None else f'feature level {level+1}'),
                     fontsize=15, y=.98)
    return figure


def attention_limits(reference, locations, image_wh):
    coordinates = np.vstack([reference[None], locations.reshape(-1, 2)])
    lower, upper = coordinates.min(0), coordinates.max(0)
    span = np.maximum(upper-lower, image_wh*.08)
    center = (lower+upper)/2
    return center-span*.7, center+span*.7


def draw_attention_samples(axis, image, reference, points, lower, upper, *, weights=None):
    axis.set_gid('attention-image')
    axis.imshow(image)
    axis.set_facecolor('#eeeeee')
    height, width = image.shape[:2]
    axis.add_patch(Rectangle((0, 0), width, height, fill=False, edgecolor='#555555', linestyle='--'))
    if weights is None:
        weights = np.ones(len(points))/len(points)
    for point_index, point in enumerate(points):
        strength = float(weights[point_index])
        opacity = .12+.88*np.sqrt(strength)
        axis.add_patch(FancyArrowPatch(reference, point, arrowstyle='->',
                                      color='#0088ff', linewidth=.35+3.65*np.sqrt(strength),
                                      alpha=opacity, mutation_scale=12,
                                      shrinkA=0, shrinkB=5))
        axis.scatter(*point, s=18+240*strength, color=SAMPLE_COLORS[point_index], edgecolors='black',
                     linewidths=.8, alpha=opacity, zorder=4)
    axis.scatter(*reference, marker='*', s=280, c='white', edgecolors='black', linewidths=1.3, zorder=6)
    axis.set(xlim=(lower[0], upper[0]), ylim=(upper[1], lower[1]))
    axis.set_aspect('equal')
    axis.set_axis_off()


def draw_attention_context(axis, image, reference, locations, image_wh):
    inset = axis.inset_axes([.01, .01, .3, .3])
    inset.imshow(image)
    all_points = np.vstack([reference[None], locations.reshape(-1, 2), [0, 0], image_wh])
    lower, upper = all_points.min(0), all_points.max(0)
    margin = (upper-lower)*.05
    inset.add_patch(Rectangle((0, 0), *image_wh, fill=False, edgecolor='black', linewidth=1))
    inset.scatter(*reference, marker='*', s=45, c='white', edgecolors='black', zorder=5)
    inset.set(xlim=(lower[0]-margin[0], upper[0]+margin[0]),
              ylim=(upper[1]+margin[1], lower[1]-margin[1]), xticks=[], yticks=[])
    inset.set_title('Image context', fontsize=7)


def draw_sample_strengths(axis, weights):
    axis.set_gid('attention-strength')
    percentages = weights*100
    axis.barh(range(len(weights)), percentages, color=SAMPLE_COLORS[:len(weights)], height=.5)
    for index, percentage in enumerate(percentages):
        if percentage == 0:
            label = '0%'
        else:
            label = f'{percentage:.2f}%' if percentage >= .01 else '<0.01%'
        axis.text(max(float(percentage), 1)+1, index, label, va='center', fontsize=9)
    axis.set(yticks=range(len(weights)), yticklabels=[f'Sample {i+1}' for i in range(len(weights))],
             xlim=(0, 108), xticks=[0, 50, 100], xlabel='Sample strength (%)')
    axis.invert_yaxis()
    axis.spines[['top', 'right']].set_visible(False)


def attention_figure(cache, image_id, steps, call_index, person, *, joint=None, head=0, layer_index=None):
    call = load_trace(cache, image_id, steps)[call_index]
    sample = sample_for(cache, image_id)
    image = np.asarray(Image.open(sample['img_path']).convert('RGB'))
    image_wh = np.array(sample['img_hw'][::-1], dtype=np.float32)
    layer_indices = list(range(len(call['layers']))) if layer_index is None else [layer_index]
    heads = list(range(cache['num_heads'])) if head is None else [head]
    if any(index < 0 or index >= len(call['layers']) for index in layer_indices):
        raise ValueError('decoder layer is out of range')
    if any(index < 0 or index >= cache['num_heads'] for index in heads):
        raise ValueError('attention head is out of range')
    selections = [(index, selected_head) for index in layer_indices for selected_head in heads]
    level_count = call['layers'][0]['weights'].shape[-2]
    heights = []
    for index, _ in selections:
        heights.extend([.45, .05] if joint is not None and index < call['num_box_layers'] else [3, 1.2])
    figure_height = max(7, sum(heights)*1.6)
    figure, axes = plt.subplots(len(heights), level_count, figsize=(4*level_count, figure_height),
                                gridspec_kw={'height_ratios': heights}, squeeze=False)
    figure.subplots_adjust(left=.06, right=.985, bottom=.85/figure_height,
                           top=1-1.3/figure_height, hspace=.4, wspace=.4)
    for selection, (index, selected_head) in enumerate(selections):
        image_axes, strength_axes = axes[selection*2:selection*2+2]
        if joint is not None and index < call['num_box_layers']:
            for axis in [*image_axes, *strength_axes]:
                axis.set_axis_off()
            image_axes[0].text(0, .4, f'Decoder layer {index+1}: box-only stage — no joint reference or joint samples.',
                               transform=image_axes[0].transAxes, fontsize=11)
            continue
        layer = call['layers'][index]
        query = query_lineage(call, index, person, joint)
        reference = layer['references'][query, :2].numpy()*image_wh
        locations = layer['locations'][query, selected_head].numpy()*image_wh
        weights = layer['weights'][query, selected_head].numpy()
        lower, upper = attention_limits(reference, locations, image_wh)
        outside = ((locations < 0) | (locations > image_wh)).any(-1)
        reference_outside = bool(((reference < 0) | (reference > image_wh)).any())
        for level, (image_axis, strength_axis) in enumerate(zip(image_axes, strength_axes)):
            draw_attention_samples(image_axis, image, reference, locations[level], lower, upper, weights=weights[level])
            image_axis.set_title(f'Decoder layer {index+1} · head {selected_head} · feature level {level+1}\n'
                                 + f'{int(outside[level].sum())}/4 samples outside image'
                                 + ('\nReference is outside image' if reference_outside else ''), fontsize=10)
            if reference_outside or outside.any():
                draw_attention_context(image_axis, image, reference, locations, image_wh)
            draw_sample_strengths(strength_axis, weights[level])
    legend = [Line2D([], [], marker='*', markersize=14, markerfacecolor='white', markeredgecolor='black',
                     linestyle='none', label='Reference point'),
              Line2D([], [], color='#0088ff', linewidth=2, marker='>', markevery=[1], label='Predicted offset')]
    legend.extend(Line2D([], [], marker='o', color=color, markeredgecolor='black', markersize=8,
                         linestyle='none', label=f'Sample {index+1}') for index, color in enumerate(SAMPLE_COLORS))
    figure.legend(handles=legend, loc='lower center', bbox_to_anchor=(.5, .005),
                  ncol=6, frameon=False, fontsize=11)
    token = 'Person box' if joint is None else JOINT_NAMES[joint].replace('_', ' ').capitalize()
    figure.suptitle(f'Reference point, predicted offsets and sample strengths\n'
                     f'{token} · image {image_id} · diffusion call {call_index+1} of {steps} · group {person}',
                     fontsize=14, y=.985)
    return figure
