"""Published cohort identities must survive reruns and incomplete downloads."""
import csv
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/benchmarks'))
import workspace_paths


class PublishedInputsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        (root/'splits').mkdir()
        self.records = {}
        rows = []
        for i, part in enumerate(('train', 'val', 'test')):
            image = root/f'case{i}.png'
            image.touch()
            rows.append({'split': part, 'id': image.name, 'patient_id': i,
                         'hva': 20+i, 'ima': 10+i})
            self.records[part] = [{'id': image.name, 'image_path': str(image),
                                  'patient_id': i, 'targets': [20+i, 10+i]}]
        with (root/'splits/hvangle.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        self.patch = patch.object(workspace_paths, 'CODE_ROOT', root)
        self.patch.start(); self.addCleanup(self.patch.stop)

    def test_complete_inputs_pass(self):
        workspace_paths.validate_paper_records('hvangle', self.records)

    def test_missing_case_is_rejected(self):
        self.records['test'] = []
        with self.assertRaises(ValueError):
            workspace_paths.validate_paper_records('hvangle', self.records)

    def test_changed_targets_and_patients_are_rejected(self):
        for field, value in [('targets', [21, 12]), ('patient_id', 500)]:
            with self.subTest(field=field):
                records = copy.deepcopy(self.records)
                records['test'][0][field] = value
                with self.assertRaises(ValueError):
                    workspace_paths.validate_paper_records('hvangle', records)

    def test_missing_image_is_rejected(self):
        Path(self.records['test'][0]['image_path']).unlink()
        with self.assertRaises(FileNotFoundError):
            workspace_paths.validate_paper_records('hvangle', self.records)


if __name__ == '__main__':
    unittest.main()
