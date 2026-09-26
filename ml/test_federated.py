from collections import OrderedDict
import unittest

import torch

from federated import IMAGENET_TRANSFORM, weighted_fedavg
from model_transport import pack_state_dict, unpack_state_dict


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

    def test_binary_model_transport_round_trip(self):
        original = OrderedDict(weight=torch.tensor([1.5, 2.5]), counter=torch.tensor([4], dtype=torch.int64))
        payload = pack_state_dict(original, ['normal', 'abnormal'], base_version=3)
        recovered, classes = unpack_state_dict(payload)
        self.assertEqual(classes, ['normal', 'abnormal'])
        self.assertTrue(torch.equal(recovered['weight'], original['weight']))
        self.assertTrue(torch.equal(recovered['counter'], original['counter']))


if __name__ == '__main__':
    unittest.main()
