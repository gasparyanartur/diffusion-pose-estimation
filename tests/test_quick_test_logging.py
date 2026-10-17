from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter
from tensorboard.backend.event_processing.event_file_loader import LegacyEventFileLoader
from scripts.utils.quick_test_logging import comparison_tag, ComparisonLogBridge


def test_comparison_tags():
    base = 'quick-test/noise_var_100/steps_3/'
    assert comparison_tag(base + 'pose_AP') == base + 'pose_AP'
    assert comparison_tag(base + 'memory_persistent/pose_AP') == base + 'pose_AP'
    assert comparison_tag(base + 'memory_reset/pose_AP') == 'quick-test-reset/noise_var_100/steps_3/pose_AP'
    assert comparison_tag(base + 'memory_reset/raw_box_AP') == 'debug/quick-test/noise_var_100/steps_3/memory_reset/raw_box_AP'
    assert comparison_tag('train/loss') == 'train/loss'


def test_bridge_append_restart_and_resume(tmp_path):
    source, target = tmp_path / 'source', tmp_path / 'target'
    writer = EventFileWriter(str(source))
    def emit(step):
        writer.add_event(Event(wall_time=100 + step, step=step, summary=Summary(value=[
            Summary.Value(tag='quick-test/noise_var_100/steps_3/memory_persistent/pose_AP', simple_value=.25)])))
        writer.flush()
    emit(1)
    bridge = ComparisonLogBridge(source, target)
    bridge.sync()
    emit(2)
    bridge.sync()
    bridge.close()
    original = {p: p.read_bytes() for p in source.glob('events*')}
    bridge = ComparisonLogBridge(source, target)
    bridge.sync()
    bridge.close()
    writer.close()
    values = [e for p in target.glob('events*') for e in LegacyEventFileLoader(str(p)).Load() if e.HasField('summary')]
    assert [(e.step, e.wall_time, e.summary.value[0].simple_value) for e in values] == [(1,101,.25),(2,102,.25)]
    assert all(e.summary.value[0].tag == 'quick-test/noise_var_100/steps_3/pose_AP' for e in values)
    assert all(p.read_bytes() == data for p, data in original.items())
    resume = source / 'resume'
    resumed_writer = EventFileWriter(str(resume))
    resumed_writer.add_event(Event(step=3, wall_time=103, summary=Summary(value=[Summary.Value(tag='train/loss', simple_value=1)])))
    resumed_writer.close()
    bridge = ComparisonLogBridge(source, target)
    bridge.sync()
    bridge.close()
    assert list((target / 'resume').glob('events*'))
