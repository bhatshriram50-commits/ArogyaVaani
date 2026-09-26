import asyncio
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from fastapi import HTTPException, UploadFile
from PIL import Image

import service


def png_bytes(color: tuple[int, int, int]) -> BytesIO:
    image = Image.new('RGB', (2, 2), color)
    output = BytesIO()
    image.save(output, format='PNG')
    output.seek(0)
    return output


class FolderDatasetUploadTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.original_root = service.DATASET_UPLOAD_ROOT
        service.DATASET_UPLOAD_ROOT = Path(self.temp_dir.name) / 'uploads'

    def tearDown(self):
        service.DATASET_UPLOAD_ROOT = self.original_root
        self.temp_dir.cleanup()

    def test_parent_folder_upload_reconstructs_normal_and_pneumonia_classes(self):
        files = [
            UploadFile(filename='chest_xray/NORMAL/normal.png', file=png_bytes((10, 20, 30))),
            UploadFile(filename='chest_xray/PNEUMONIA/pneumonia.png', file=png_bytes((30, 20, 10))),
        ]

        result = asyncio.run(service.upload_dataset(files=files))
        dataset_dir = Path(result['dataset_dir'])

        self.assertEqual(result['dataset_name'], 'chest_xray')
        self.assertEqual(result['storage'], 'local-only')
        self.assertEqual(result['validation']['classes'], ['NORMAL', 'PNEUMONIA'])
        self.assertEqual(result['validation']['class_distribution'], {'NORMAL': 1, 'PNEUMONIA': 1})
        self.assertTrue((dataset_dir / 'NORMAL' / 'normal.png').is_file())
        self.assertTrue((dataset_dir / 'PNEUMONIA' / 'pneumonia.png').is_file())
        self.assertFalse((dataset_dir / 'chest_xray').exists())

    def test_upload_rejects_unsupported_files(self):
        files = [
            UploadFile(filename='chest_xray/NORMAL/readme.txt', file=BytesIO(b'not an image')),
            UploadFile(filename='chest_xray/PNEUMONIA/pneumonia.png', file=png_bytes((30, 20, 10))),
        ]

        with self.assertRaises(HTTPException) as error:
            asyncio.run(service.upload_dataset(files=files))

        self.assertEqual(error.exception.status_code, 400)
        self.assertIn('only JPG, JPEG, PNG, BMP, TIFF, or WEBP images', error.exception.detail)

    def test_upload_rejects_invalid_images_before_training(self):
        files = [
            UploadFile(filename='chest_xray/NORMAL/broken.png', file=BytesIO(b'not a PNG')),
            UploadFile(filename='chest_xray/PNEUMONIA/pneumonia.png', file=png_bytes((30, 20, 10))),
        ]

        with self.assertRaises(HTTPException) as error:
            asyncio.run(service.upload_dataset(files=files))

        self.assertEqual(error.exception.status_code, 400)
        self.assertIn('invalid image', error.exception.detail)


if __name__ == '__main__':
    unittest.main()
