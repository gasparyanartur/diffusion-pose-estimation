import argparse
import hashlib
import time
from pathlib import Path

from tensorboard.backend.event_processing.event_file_loader import LegacyEventFileLoader
from tensorboard.summary.writer.event_file_writer import EventFileWriter

COMPARISON_METRICS = frozenset({
    'pose_AP', 'bbox_AP', 'pose_AP50', 'pose_AP75', 'pose_AR', 'oracle_oks', 'coverage75',
})


def comparison_tag(tag: str) -> str:
    if not tag.startswith('quick-test/'):
        return tag
    parts = tag.split('/')
    metric = parts[-1]
    condition = parts[1:-1]
    if metric not in COMPARISON_METRICS:
        return 'debug/' + tag
    prefix = 'quick-test'
    if condition and condition[-1] == 'memory_reset':
        prefix = 'quick-test-reset'
        condition = condition[:-1]
    elif condition and condition[-1] == 'memory_persistent':
        condition = condition[:-1]
    return '/'.join([prefix, *condition, metric])


class ComparisonLogBridge:
    def __init__(self, source: Path, target: Path):
        self.source = Path(source).resolve()
        self.target = Path(target).resolve()
        if self.source == self.target or self.source in self.target.parents or self.target in self.source.parents:
            raise ValueError('Source and comparison log directories must be separate')
        self.loaders = {}
        self.writers = {}
        self.seen = {}

    def _writer(self, relative_directory):
        if relative_directory not in self.writers:
            directory = self.target / relative_directory
            fingerprints = set()
            for path in sorted(directory.glob('events.out.tfevents.*')):
                for event in LegacyEventFileLoader(str(path)).Load():
                    if event.HasField('summary'):
                        fingerprints.add(hashlib.sha256(event.SerializeToString()).digest())
            self.seen[relative_directory] = fingerprints
            self.writers[relative_directory] = EventFileWriter(str(directory))
        return self.writers[relative_directory]

    def sync(self):
        copied = 0
        for path in sorted(self.source.rglob('events.out.tfevents.*')):
            relative_directory = path.parent.relative_to(self.source)
            writer = self._writer(relative_directory)
            loader = self.loaders.setdefault(path, LegacyEventFileLoader(str(path)))
            for event in loader.Load():
                if not event.HasField('summary'):
                    continue
                for value in event.summary.value:
                    value.tag = comparison_tag(value.tag)
                fingerprint = hashlib.sha256(event.SerializeToString()).digest()
                if fingerprint in self.seen[relative_directory]:
                    continue
                writer.add_event(event)
                self.seen[relative_directory].add(fingerprint)
                copied += 1
            writer.flush()
        return copied

    def close(self):
        for writer in self.writers.values():
            writer.close()


def main():
    parser = argparse.ArgumentParser(description='Maintain a comparison view of TensorBoard summaries without changing source logs.')
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--interval', type=float, default=10)
    parser.add_argument('--once', action='store_true')
    arguments = parser.parse_args()
    if arguments.interval <= 0:
        parser.error('--interval must be positive')
    bridge = ComparisonLogBridge(arguments.source, arguments.target)
    try:
        while True:
            copied = bridge.sync()
            if copied:
                print(f'Copied {copied} new summary events', flush=True)
            if arguments.once:
                break
            time.sleep(arguments.interval)
    finally:
        bridge.close()


if __name__ == '__main__':
    main()
