"""Square CLIP-tile geometry. No CLIP checkpoint, no FEA."""

from __future__ import annotations

import unittest

import torch

from guidance.clip_scales import (
    building_storey_centers,
    layout_for_structure,
    parse_clip_scales,
    scale_boxes_to_image,
    scales_token,
)


class ParseScalesTest(unittest.TestCase):

    def test_gme_and_aliases(self):
        self.assertEqual(parse_clip_scales('g,m,e'), ('g', 'm', 'e'))
        self.assertEqual(parse_clip_scales('gme'), ('g', 'm', 'e'))
        self.assertEqual(parse_clip_scales('m'), ('m',))
        self.assertEqual(scales_token(('g', 'm', 'e')), 'gme')

    def test_rejects_unknown(self):
        with self.assertRaises(ValueError):
            parse_clip_scales('storey')


class BuildingTilesTest(unittest.TestCase):

    def test_tall_has_one_centered_square_per_storey(self):
        layout = layout_for_structure(256, 128, kind='building', interval=64)
        self.assertEqual(layout.module_width, 64)
        self.assertEqual(layout.module_height, 64)
        self.assertEqual(int(layout.module.shape[0]), 4)
        self.assertTrue(torch.all(layout.module[:, 0] == 64))
        self.assertTrue(torch.all(layout.module[:, 2] == 64))
        self.assertTrue(torch.all(layout.module[:, 3] == 64))
        self.assertEqual(building_storey_centers(256, 64), [32.0, 96.0, 160.0, 224.0])

    def test_short_has_two_centered_squares_per_storey(self):
        layout = layout_for_structure(150, 300, kind='building', interval=50)
        self.assertEqual(layout.module_width, 50)
        self.assertEqual(layout.module_height, 50)
        self.assertEqual(int(layout.module.shape[0]), 6)
        torch.testing.assert_close(
            layout.module[:2, 0], torch.tensor([75.0, 225.0]))
        self.assertEqual(layout.element_side, 12)
        self.assertEqual(int(layout.element.shape[0]), 6 * 4)

    def test_element_windows_are_square_and_inside_the_grid(self):
        layout = layout_for_structure(256, 128, kind='building', interval=64)
        half = layout.element_side / 2.0
        self.assertTrue(torch.all(layout.element[:, 2] == layout.element[:, 3]))
        self.assertGreaterEqual(float(layout.element[:, 0].min() - half), -1e-4)
        self.assertLessEqual(float(layout.element[:, 0].max() + half), 128 + 1e-4)
        self.assertGreaterEqual(float(layout.element[:, 1].min() - half), -1e-4)
        self.assertLessEqual(float(layout.element[:, 1].max() + half), 256 + 1e-4)


class BridgeTilesTest(unittest.TestCase):

    def test_panels_are_full_depth_squares(self):
        layout = layout_for_structure(72, 448, kind='bridge', interval=72)
        self.assertEqual(layout.kind, 'bridge')
        self.assertEqual(layout.module_width, 72)
        self.assertEqual(layout.module_height, 72)
        self.assertEqual(int(layout.module.shape[0]), 448 // 72)
        self.assertTrue(torch.all(layout.module[:, 1] == 36.0))
        self.assertTrue(torch.all(layout.module[:, 2] == 72))
        self.assertTrue(torch.all(layout.module[:, 3] == 72))

    def test_scale_to_image_preserves_aspect(self):
        layout = layout_for_structure(72, 448, kind='bridge', interval=72)
        scaled = scale_boxes_to_image(
            layout.module,
            src_height=72,
            src_width=448,
            dst_height=72 * 4,
            dst_width=448 * 4,
        )
        self.assertTrue(torch.allclose(scaled[:, 2], scaled[:, 3]))
        self.assertTrue(torch.allclose(scaled[:, 2], torch.full((scaled.shape[0],), 288.0)))


if __name__ == '__main__':
    unittest.main()
