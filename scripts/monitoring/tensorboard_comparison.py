import argparse
import hashlib
import json
from pathlib import Path
import signal
import time

from tensorboard.backend.event_processing.event_file_loader import LegacyEventFileLoader
from tensorboard.compat.proto import event_pb2, summary_pb2
from tensorboard.summary.writer.event_file_writer import EventFileWriter


EXPERIMENT_PREFIXES = {
    'gradient_balanced_residual', 'proposal_attention_residual', 'token_age_diffusion',
    'ordered_birth_only', 'ordered_coordinate', 'ordered_sequence', 'progressive',
    'sampler_comparison',
}


def metric_destination(run, tag):
    if tag.startswith('quick-test/'):
        parts = tag.split('/')
        configuration, metric = parts[1:-1], parts[-1]
        if run.startswith('V46-sampler-') and configuration[0:1] == ['sampler_comparison']:
            return None  # These runs also contain the original sampler-specific measurements.
        if run.startswith('V51-birth-final-AP'):
            if 'birth' in configuration or run.endswith('/birth'):
                return 'V51-gradient-balanced-residual/birth', f'quick-test/{metric}'
            return None  # The main trainer already supplies the full final-output curve.
        if not configuration:
            label = run if run.endswith(('/birth', '/final')) else f'{run}/final'
        elif configuration[0].startswith(('noise_var_', 'noise_')):
            label = '/'.join((run, *configuration))
        elif configuration[0] in EXPERIMENT_PREFIXES:
            stage = 'birth' if configuration[-1] == 'birth' else 'final'
            sampling = [value for value in configuration[1:]
                        if value not in ('birth', 'final') and not value.startswith(('rows_', 'stages_'))]
            label = '/'.join((run, stage, *sampling))
        else:
            raise ValueError(f'Unknown quick-test configuration: {tag}')
        return label, f'quick-test/{metric}'
    prefix, _, suffix = tag.partition('/')
    if prefix in EXPERIMENT_PREFIXES:
        return run, f'diagnostics/{suffix}'
    if tag in ('val/proposal_attention_selection_AP', 'val/gradient_balanced_selection_AP'):
        return run, 'val/selection_AP'
    return run, tag


class ComparisonView:
    def __init__(self, sources, output):
        self.sources = {label: Path(path) for label, path in sources.items()}
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.state_path = self.output/'cursors.json'
        self.cursors = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        self.loaders, self.read_counts, self.signatures, self.writers = {}, {}, {}, {}
        self.routes = {}
        self.fingerprints = {}

    def _seen(self, label, event):
        if label not in self.fingerprints:
            fingerprints = set()
            for path in (self.output/'events'/label).glob('events.out.tfevents.*'):
                for previous in LegacyEventFileLoader(str(path)).Load():
                    if previous.HasField('summary'):
                        fingerprints.add(hashlib.sha256(previous.SerializeToString()).digest())
            self.fingerprints[label] = fingerprints
        checksum = hashlib.sha256(event.SerializeToString()).digest()
        if checksum in self.fingerprints[label]:
            return True
        self.fingerprints[label].add(checksum)
        return False

    def _writer(self, label):
        if label not in self.writers:
            self.writers[label] = EventFileWriter(str(self.output/'events'/label))
        return self.writers[label]

    def _write(self, run, event):
        groups = {}
        for value in event.summary.value:
            destination = metric_destination(run, value.tag)
            if destination is None:
                continue
            label, tag = destination
            origin = (run, value.tag)
            key = (label, tag)
            if key in self.routes and self.routes[key] != origin:
                raise ValueError(f'Ambiguous metric mapping: {key}: {self.routes[key]} and {origin}')
            self.routes[key] = origin
            normalized = summary_pb2.Summary.Value()
            normalized.CopyFrom(value)
            normalized.tag = tag
            if normalized.metadata.display_name == value.tag:
                normalized.metadata.display_name = tag
            groups.setdefault(label, []).append(normalized)
        written = 0
        for label, values in groups.items():
            normalized_event = event_pb2.Event(wall_time=event.wall_time,
                step=event.step, summary=summary_pb2.Summary(value=values))
            if not self._seen(label, normalized_event):
                self._writer(label).add_event(normalized_event)
                written += len(values)
        return written

    def sync(self):
        written = 0
        for label, root in self.sources.items():
            for path in sorted(root.rglob('events.out.tfevents.*')):
                information = path.stat()
                key = f'{label}:{path}'
                signature = (information.st_size, information.st_mtime_ns)
                if self.signatures.get(key) == signature:
                    continue
                if key not in self.loaders:
                    self.loaders[key] = LegacyEventFileLoader(str(path))
                    self.read_counts[key] = 0
                relative = path.parent.relative_to(root)
                run = label if str(relative) == '.' else f'{label}/{relative}'
                for event in self.loaders[key].Load():
                    self.read_counts[key] += 1
                    if self.read_counts[key] <= self.cursors.get(key, 0):
                        continue
                    if event.HasField('summary'):
                        written += self._write(run, event)
                    elif not event.file_version:
                        self._writer(run).add_event(event)
                self.cursors[key] = self.read_counts[key]
                self.signatures[key] = signature
            (self.output/'status.json').write_text(json.dumps(dict(
                phase='backfilling', current_source=label, written_values=written,
                source_roots=len(self.sources), output_runs=len(self.writers), last_sync=time.time()))+'\n')
        for writer in self.writers.values():
            writer.flush()
        temporary = self.state_path.with_suffix('.new')
        temporary.write_text(json.dumps(self.cursors, indent=2)+'\n')
        temporary.replace(self.state_path)
        return written

    def close(self):
        for writer in self.writers.values():
            writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--watch', action='store_true')
    arguments = parser.parse_args()
    sources = json.loads(arguments.sources.read_text())
    view = ComparisonView(sources, arguments.output)
    running = True

    def stop(signum, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        while running:
            count = view.sync()
            (arguments.output/'status.json').write_text(json.dumps(dict(
                phase='watching' if arguments.watch else 'complete', last_sync=time.time(),
                written_values=count, source_roots=len(sources), output_runs=len(view.writers)))+'\n')
            print(f'Canonical comparison sync: {count} values', flush=True)
            if not arguments.watch:
                break
            for _ in range(30):
                if not running:
                    break
                time.sleep(1)
    finally:
        view.close()


if __name__ == '__main__':
    main()
