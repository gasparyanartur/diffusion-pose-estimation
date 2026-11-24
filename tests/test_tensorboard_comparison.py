from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.compat.proto import event_pb2, summary_pb2
from tensorboard.summary.writer.event_file_writer import EventFileWriter

from scripts.monitoring.tensorboard_comparison import ComparisonView, metric_destination


def test_equivalent_metrics_share_tags_and_configuration_moves_to_run_label():
    assert metric_destination('V50', 'quick-test/proposal_attention_residual/rows_50/pose_AP') == (
        'V50/final', 'quick-test/pose_AP')
    assert metric_destination('V51', 'quick-test/gradient_balanced_residual/rows_50/pose_AP') == (
        'V51/final', 'quick-test/pose_AP')
    assert metric_destination('V5', 'quick-test/noise_var_100/steps_4/bbox_AP') == (
        'V5/noise_var_100/steps_4', 'quick-test/bbox_AP')
    assert metric_destination('V51-birth-final-AP',
        'quick-test/gradient_balanced_residual/rows_50/birth/pose_AP') == (
        'V51-gradient-balanced-residual/birth', 'quick-test/pose_AP')
    assert metric_destination('V51-birth-final-AP',
        'quick-test/gradient_balanced_residual/rows_50/final/pose_AP') is None
    for run, prefix in (('V50', 'proposal_attention_residual'), ('V51', 'gradient_balanced_residual')):
        assert metric_destination(run, f'{prefix}/teacher_forced/rank_01/birth_geometry') == (
            run, 'diagnostics/teacher_forced/rank_01/birth_geometry')
    assert metric_destination('V51', 'train/total_step') == ('V51', 'train/total_step')
    assert metric_destination('V46-sampler-DDIM-epoch046', 'quick-test/sampler_comparison/rows_64/pose_AP') is None
    assert metric_destination('V44', 'quick-test/noise_10/input_latent_mse') == (
        'V44/noise_10', 'quick-test/input_latent_mse')


def test_backfill_is_exact_incremental_and_restart_idempotent(tmp_path):
    source = tmp_path/'source'
    writer = EventFileWriter(str(source))

    def emit(step, value):
        writer.add_event(event_pb2.Event(step=step, wall_time=100.+step,
            summary=summary_pb2.Summary(value=[summary_pb2.Summary.Value(
                tag='quick-test/gradient_balanced_residual/rows_50/pose_AP', simple_value=value)])))
        writer.flush()

    emit(4, .25)
    output = tmp_path/'view'
    view = ComparisonView({'V51': str(source)}, output)
    assert view.sync() == 1
    assert view.sync() == 0
    emit(8, .5)
    assert view.sync() == 1
    view.close()
    view = ComparisonView({'V51': str(source)}, output)
    assert view.sync() == 0
    view.close()
    (output/'cursors.json').unlink()  # Recover a crash between event flush and cursor persistence.
    view = ComparisonView({'V51': str(source)}, output)
    assert view.sync() == 0
    view.close()
    events = EventAccumulator(str(output/'events/V51/final'), size_guidance={'scalars': 0}).Reload()
    points = events.Scalars('quick-test/pose_AP')
    assert [(point.step, point.wall_time, point.value) for point in points] == [(4, 104., .25), (8, 108., .5)]
    assert 'gradient_balanced_residual' not in str(events.Tags())
    assert len(list(source.glob('events*'))) == 1
    writer.close()
