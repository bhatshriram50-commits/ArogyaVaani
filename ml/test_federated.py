from collections import OrderedDict
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from federated import IMAGENET_TRANSFORM, weighted_fedavg
from model_transport import manifest_from_payload, pack_state_dict, unpack_state_dict
from medical_tasks import TASKS, get_task, task_checkpoint
from service import checkpoint_for_task, select_inference_checkpoint, validate_checkpoint_metadata
import service


class FederatedTests(unittest.TestCase):
    def test_weighted_fedavg_uses_sample_counts(self):
        first = OrderedDict(weight=torch.tensor([1.0, 3.0]))
        second = OrderedDict(weight=torch.tensor([5.0, 7.0]))
        result = weighted_fedavg([(first, 1), (second, 3)])
        self.assertTrue(torch.allclose(result['weight'], torch.tensor([4.0, 6.0])))

    def test_weighted_fedavg_rejects_tensor_shape_mismatch(self):
        first = OrderedDict(weight=torch.tensor([1.0, 3.0]))
        second = OrderedDict(weight=torch.tensor([5.0]))
        with self.assertRaisesRegex(ValueError, 'shape mismatch'):
            weighted_fedavg([(first, 1), (second, 1)])

    def test_preprocessing_has_imagenet_normalization(self):
        normalize = IMAGENET_TRANSFORM.transforms[-1]
        self.assertEqual(tuple(normalize.mean), (0.485, 0.456, 0.406))
        self.assertEqual(tuple(normalize.std), (0.229, 0.224, 0.225))

    def test_local_training_reports_accuracy_for_each_class(self):
        from PIL import Image
        from federated import local_train
        with TemporaryDirectory() as temporary:
            dataset_dir = Path(temporary) / 'dataset'
            for label, color in (('NORMAL', (20, 20, 20)), ('PNEUMONIA', (230, 230, 230))):
                class_dir = dataset_dir / label
                class_dir.mkdir(parents=True)
                Image.new('RGB', (16, 16), color=color).save(class_dir / 'image.png')
            _, metrics = local_train(str(dataset_dir), num_classes=2, epochs=1, batch_size=2)
            self.assertEqual(set(metrics['class_accuracy']), {'NORMAL', 'PNEUMONIA'})
            self.assertEqual({values['total'] for values in metrics['class_accuracy'].values()}, {1})

    def test_binary_model_transport_round_trip(self):
        original = OrderedDict(weight=torch.tensor([1.5, 2.5]), counter=torch.tensor([4], dtype=torch.int64))
        payload = pack_state_dict(original, ['normal', 'abnormal'], base_version=3)
        recovered, classes = unpack_state_dict(payload)
        self.assertEqual(classes, ['normal', 'abnormal'])
        self.assertTrue(torch.equal(recovered['weight'], original['weight']))
        self.assertTrue(torch.equal(recovered['counter'], original['counter']))

    def test_task_configuration_and_transport_are_task_specific(self):
        self.assertEqual(set(TASKS), {'PNEUMONIA', 'RETINOPATHY', 'BRAIN_TUMOR'})
        self.assertEqual(get_task('RETINOPATHY').modality, 'FUNDUS_IMAGE')
        self.assertIn('retinopathy', task_checkpoint('RETINOPATHY', 'latest.pt'))
        with self.assertRaises(ValueError):
            get_task('UNKNOWN')
        payload = pack_state_dict(OrderedDict(weight=torch.tensor([1.0])), ['normal', 'abnormal'], task='BRAIN_TUMOR')
        self.assertEqual(manifest_from_payload(payload)['task'], 'BRAIN_TUMOR')

    def test_task_configuration_is_kept_consistent_across_service_boundaries(self):
        root = Path(__file__).resolve().parents[1]
        frontend = (root / 'frontend' / 'src' / 'medicalTasks.ts').read_text(encoding='utf-8')
        backend = (root / 'backend' / 'src' / 'medicalTasks.ts').read_text(encoding='utf-8')
        for task in TASKS.values():
            self.assertIn(f"'{task.task_id}'", frontend)
            self.assertIn(f"{task.task_id}:", backend)
            self.assertIn(task.display_name, frontend)
            self.assertEqual(task.architecture, 'densenet121')
            self.assertEqual(task.input_size, (224, 224))
            self.assertEqual(task.preprocessing, 'imagenet-rgb')

    def test_binary_transport_rejects_non_finite_tensors(self):
        with self.assertRaisesRegex(ValueError, 'NaN or infinite'):
            pack_state_dict(OrderedDict(weight=torch.tensor([float('nan')])), ['normal', 'abnormal'])

    def test_task_checkpoint_and_metadata_cannot_cross_task_boundaries(self):
        with self.assertRaisesRegex(ValueError, 'Checkpoint paths'):
            checkpoint_for_task('PNEUMONIA', 'local_models/retinopathy/latest.pt', 'latest.pt')
        valid = {'task': 'RETINOPATHY', 'architecture': 'densenet121', 'classes': ['Mild', 'Moderate', 'No_DR', 'Proliferate_DR', 'Severe'], 'state_dict': {'weight': torch.tensor([1.0])}}
        self.assertEqual(validate_checkpoint_metadata(valid, 'RETINOPATHY'), ['Mild', 'Moderate', 'No_DR', 'Proliferate_DR', 'Severe'])
        with self.assertRaisesRegex(ValueError, 'does not match'):
            validate_checkpoint_metadata(valid, 'PNEUMONIA')

    def test_inference_does_not_use_a_version_zero_bootstrap_over_local_training(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_root = root / 'models'
            model_dir = model_root / 'pneumonia'
            model_dir.mkdir(parents=True)
            common = {'task': 'PNEUMONIA', 'architecture': 'densenet121', 'classes': ['NORMAL', 'PNEUMONIA'], 'state_dict': {'weight': torch.tensor([1.0])}}
            torch.save({**common, 'version': '0', 'bootstrap': True}, model_dir / 'global.pt')
            torch.save({**common, 'base_version': 0}, model_dir / 'latest.pt')
            previous = os.environ.get('LOCAL_MODEL_ROOT')
            os.environ['LOCAL_MODEL_ROOT'] = str(model_root)
            try:
                checkpoint, _, _ = select_inference_checkpoint('PNEUMONIA')
                self.assertEqual(checkpoint.name, 'latest.pt')
                torch.save({**common, 'version': '1'}, model_dir / 'global.pt')
                checkpoint, _, _ = select_inference_checkpoint('PNEUMONIA')
                self.assertEqual(checkpoint.name, 'global.pt')
            finally:
                if previous is None:
                    del os.environ['LOCAL_MODEL_ROOT']
                else:
                    os.environ['LOCAL_MODEL_ROOT'] = previous

    def test_pending_avm1_is_durable_until_a_coordinator_confirms_submission(self):
        with TemporaryDirectory() as temporary:
            original_root = service.LOCAL_UPDATE_ROOT
            try:
                service.LOCAL_UPDATE_ROOT = Path(temporary)
                payload = pack_state_dict(OrderedDict(weight=torch.tensor([1.0])), ['normal', 'abnormal'], task='PNEUMONIA')
                service.persist_pending_update('PNEUMONIA', 7, payload, {'samples': 3, 'base_global_model_version': 0})
                persisted, metadata = service.read_pending_update('PNEUMONIA', 7)
                self.assertEqual(persisted, payload)
                self.assertEqual(metadata['samples'], 3)
                self.assertTrue(service.pending_update_path('PNEUMONIA', 7).is_file())
                # Recreating the in-memory job registry (as a logout/reload
                # does) has no effect on the artifact or its retry metadata.
                service.training_jobs.clear()
                self.assertEqual(service.read_pending_update('PNEUMONIA', 7)[0], payload)
                service.delete_pending_update('PNEUMONIA', 7)
                self.assertFalse(service.pending_update_path('PNEUMONIA', 7).exists())
            finally:
                service.LOCAL_UPDATE_ROOT = original_root

    def test_end_to_end_serialization_and_deserialization_precision(self):
        """Test B: Parameter serialization and reconstruction precision."""
        original = OrderedDict([
            ('features.conv0.weight', torch.randn(64, 3, 7, 7, dtype=torch.float32)),
            ('features.norm0.bias', torch.randn(64, dtype=torch.float32)),
            ('classifier.weight', torch.randn(2, 1024, dtype=torch.float32)),
            ('classifier.bias', torch.tensor([0.1234567, -0.9876543], dtype=torch.float32)),
        ])
        classes = ['normal', 'pneumonia']
        payload = pack_state_dict(original, classes, architecture='densenet121', base_version=1, task='PNEUMONIA')
        reconstructed, unpacked_classes = unpack_state_dict(payload)
        self.assertEqual(unpacked_classes, classes)
        self.assertEqual(list(reconstructed.keys()), list(original.keys()))
        for key in original:
            self.assertEqual(reconstructed[key].shape, original[key].shape)
            self.assertEqual(reconstructed[key].dtype, original[key].dtype)
            self.assertTrue(torch.allclose(reconstructed[key], original[key], atol=1e-7))

    def test_densenet_bootstrap_architecture_and_classes(self):
        """Test A: DenseNet model architecture initialization."""
        from federated import create_densenet
        model1 = create_densenet(2)
        model2 = create_densenet(2)
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            self.assertTrue(torch.equal(p1, p2))
        self.assertEqual(model1.classifier.out_features, 2)

    def test_checkpoint_classifier_shape_mismatch_raises_clear_error(self):
        from federated import local_train
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            # Create a 5-class checkpoint (e.g. Retinopathy)
            ckpt_file = temp_path / 'retinopathy_ckpt.pt'
            densenet5 = torch.nn.Linear(1024, 5)
            torch.save({'state_dict': {'classifier.weight': torch.randn(5, 1024), 'classifier.bias': torch.randn(5)}}, ckpt_file)
            # Create dummy dataset directory with 2 classes (e.g. Pneumonia)
            dataset_dir = temp_path / 'dataset'
            (dataset_dir / 'NORMAL').mkdir(parents=True)
            (dataset_dir / 'PNEUMONIA').mkdir(parents=True)
            from PIL import Image
            img = Image.new('RGB', (10, 10), color='white')
            img.save(dataset_dir / 'NORMAL' / 'img1.png')
            img.save(dataset_dir / 'PNEUMONIA' / 'img2.png')

            with self.assertRaisesRegex(ValueError, 'output classes.*does not match'):
                local_train(str(dataset_dir), num_classes=2, epochs=1, initial_checkpoint=str(ckpt_file))

    def test_all_three_diseases_properly_configured(self):
        expected_tasks = {
            'PNEUMONIA': ('Pneumonia', 'LUNG', 'CHEST_XRAY'),
            'RETINOPATHY': ('Retinopathy', 'EYE', 'FUNDUS_IMAGE'),
            'BRAIN_TUMOR': ('Brain Tumor', 'BRAIN', 'MRI'),
        }
        for task_id, (display_name, organ, modality) in expected_tasks.items():
            task = get_task(task_id)
            self.assertEqual(task.display_name, display_name)
            self.assertEqual(task.organ, organ)
            self.assertEqual(task.modality, modality)
            self.assertTrue(task.enabled)
            self.assertEqual(task.architecture, 'densenet121')
    def test_url_request_is_not_shadowed_by_fastapi_request(self):
        from service import UrlRequest
        req = UrlRequest('http://127.0.0.1:4000/api/test', data=b'payload', method='POST')
        self.assertEqual(req.data, b'payload')
        self.assertEqual(req.get_method(), 'POST')


if __name__ == '__main__':
    unittest.main()


