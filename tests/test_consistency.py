"""Tests for the drift audit and the best-of-n selector.

Everything is synthesised with ImageDraw rather than fixtured: the whole module
is arithmetic over pixels, so a test that draws the pixels can assert on exact
numbers -- a cream outline must come back as exactly (255,245,220), a 2px ring
must measure 2.00 thick. No network is touched anywhere in consistency.py, so
there is nothing to monkeypatch; that is a property worth keeping, not an
oversight.
"""

import os
import tempfile
import unittest

from PIL import Image, ImageDraw

from devgraphics import consistency

ORANGE = (255, 122, 40, 255)
CREAM = (255, 245, 220, 255)
BLUE = (32, 80, 192, 255)
PALETTE = ["#FF7A28", "#FFC400", "#E8483A", "#FFF5DC"]


def disc(radius, fill=ORANGE, outline=CREAM, width=6, size=128):
    """A filled circle with an outline: the shape this project actually makes."""
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    centre = size // 2
    ImageDraw.Draw(im).ellipse(
        [centre - radius, centre - radius, centre + radius, centre + radius],
        fill=fill, outline=outline, width=width)
    return im


def ring(width, radius=100, size=256):
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    centre = size // 2
    ImageDraw.Draw(im).ellipse(
        [centre - radius, centre - radius, centre + radius, centre + radius],
        outline=ORANGE, width=width)
    return im


def consistent_set():
    """Eight on-style icons, one off-palette, one filling the frame."""
    icons = dict(("ok%d" % i, disc(30 + i * 2)) for i in range(8))
    icons["offpalette"] = disc(36, fill=BLUE)
    icons["fullframe"] = disc(68)
    return icons


class Features(unittest.TestCase):

    def test_six_features_over_opaque_pixels_only(self):
        feats = consistency.features(disc(40))
        self.assertEqual(sorted(feats),
                         ["hist", "ink", "lum", "outline", "sat", "thick"])
        # 4x4x4 histogram, normalised over the opaque pixels
        self.assertEqual(len(feats["hist"]), 64)
        self.assertAlmostEqual(sum(feats["hist"]), 1.0, places=6)
        self.assertLess(feats["ink"], 1.0)
        # the two drawn colours, and nothing else
        self.assertEqual(len([v for v in feats["hist"] if v]), 2)

    def test_outline_colour_is_recovered_exactly(self):
        self.assertEqual(consistency.features(disc(40))["outline"], (255, 245, 220))

    def test_thickness_tracks_stroke_width_and_separates_solids(self):
        thin = consistency.features(ring(2))["thick"]
        thick = consistency.features(ring(16))["thick"]
        self.assertAlmostEqual(thin, 2.0, places=1)
        self.assertAlmostEqual(thick, 12.5, delta=0.5)      # measured 12.47
        # the same circle filled in reads roughly its radius instead -- an order
        # of magnitude clear of the stroke, which is what separates line art
        # from a solid fill
        solid = consistency.features(
            disc(100, outline=None, width=0, size=256))["thick"]
        self.assertGreater(solid, 0.7 * 100)        # measured 79.2 at r=100
        self.assertLess(solid, 1.0 * 100)
        self.assertGreater(solid, 5 * thick)

    def test_subject_touching_the_canvas_edge_still_has_a_boundary(self):
        """PIL's MinFilter copies the border instead of eroding it, so without
        the transparent pad the perimeter would be 0 and this would divide."""
        full = consistency.features(Image.new("RGBA", (128, 128), ORANGE))
        self.assertEqual(full["ink"], 1.0)
        self.assertAlmostEqual(full["thick"], 2.0 * 128 * 128 / (4 * 128 - 4),
                               places=6)

    def test_blank_render_has_no_features(self):
        self.assertIsNone(
            consistency.features(Image.new("RGBA", (64, 64), (0, 0, 0, 0))))

    def test_path_and_image_agree(self):
        icon = disc(40)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "flame.png")
            icon.save(path)
            self.assertEqual(consistency.features(path),
                             consistency.features(icon))


class Audit(unittest.TestCase):

    def test_outliers_are_flagged_and_the_consistent_ones_are_not(self):
        result = consistency.audit(consistent_set())
        self.assertTrue(result["gated"])
        self.assertEqual(sorted(result["flagged"]), ["fullframe", "offpalette"])
        # the off-palette icon is caught by the histogram, not by luck
        self.assertIn("palette", [f[0] for f in result["flagged"]["offpalette"]])
        self.assertIn("ink", [f[0] for f in result["flagged"]["fullframe"]])
        for name in result["scores"]:
            if name.startswith("ok"):
                for key, z in result["scores"][name].items():
                    self.assertLessEqual(abs(z), consistency.Z_CUTOFF,
                                         "%s drifted on %s" % (name, key))

    def test_background_floor_is_absolute_and_survives_the_gate_being_off(self):
        icons = {"a": disc(34), "b": disc(36), "c": disc(38)}
        result = consistency.audit(icons, bg_shares={"a": 0.34, "b": 0.75})
        self.assertFalse(result["gated"])            # only three icons
        self.assertEqual(sorted(result["flagged"]), ["a"])
        key, value, z = result["flagged"]["a"][0]
        self.assertEqual((key, value, z), ("bg_share", 0.34, None))

    def test_small_set_says_so_instead_of_returning_nothing(self):
        icons = {"a": disc(30), "b": disc(34), "c": disc(60), "d": disc(32)}
        result = consistency.audit(icons)
        self.assertFalse(result["gated"])
        self.assertEqual(result["flagged"], {})
        self.assertEqual(len(result["values"]), 4)   # scored, just not gated
        self.assertTrue(any("relative gate is OFF" in n for n in result["notes"]))
        text = consistency.report(result)
        self.assertIn("[relative gate OFF]", text)
        # the promise the note makes: numbers instead of an empty result
        for name in icons:
            self.assertIn("  %-16s" % name, text)

    def test_mad_of_zero_does_not_divide_by_zero(self):
        """Eight identical icons plus one bad one: MAD is 0, and returning zeros
        there would go blind on exactly the case the audit exists for."""
        icons = dict(("same%d" % i, disc(40)) for i in range(8))
        icons["odd"] = disc(58)
        result = consistency.audit(icons)
        self.assertEqual(result["stats"]["ink"]["kind"], "MeanAD")
        self.assertGreater(result["stats"]["ink"]["spread"], 0.0)
        self.assertEqual(sorted(result["flagged"]), ["odd"])
        self.assertTrue(any("MAD was 0" in n for n in result["notes"]))

    def test_a_set_with_no_spread_at_all_flags_nothing(self):
        icons = dict(("same%d" % i, disc(40)) for i in range(9))
        result = consistency.audit(icons)
        self.assertEqual(result["flagged"], {})
        for key in consistency.KEYS:
            self.assertEqual(result["stats"][key]["kind"], "none")
            self.assertEqual(result["stats"][key]["spread"], 0.0)

    def test_blank_render_is_flagged_not_scored(self):
        icons = consistent_set()
        icons["empty"] = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
        result = consistency.audit(icons)
        self.assertIn("empty", result["flagged"])
        self.assertEqual(result["flagged"]["empty"][0][0], "blank")
        self.assertNotIn("empty", result["values"])
        self.assertEqual(result["count"], 11)
        self.assertEqual(result["scored"], 10)

    def test_a_sequence_of_paths_is_named_by_filename_stem(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for name, icon in sorted(consistent_set().items()):
                path = os.path.join(tmp, name + ".png")
                icon.save(path)
                paths.append(path)
            result = consistency.audit(paths)
        self.assertEqual(sorted(result["flagged"]), ["fullframe", "offpalette"])


class Report(unittest.TestCase):

    def test_names_each_flagged_icon_its_feature_and_its_z(self):
        text = consistency.report(consistency.audit(consistent_set()))
        self.assertIn("DRIFT  offpalette", text)
        self.assertIn("palette", text)
        self.assertIn("z=", text)
        self.assertIn("8/10 within tolerance", text)

    def test_states_the_limitation_in_the_output_not_only_the_docs(self):
        text = consistency.report(consistency.audit(consistent_set()))
        self.assertIn("WRONG OBJECT", text)
        self.assertIn("not semantics", text)
        self.assertIn("does not catch subject failure", text)

    def test_is_ascii_for_a_windows_console(self):
        result = consistency.audit(consistent_set(), bg_shares={"fullframe": 0.34})
        consistency.report(result).encode("ascii")   # raises if it is not


class Nearest(unittest.TestCase):

    def test_picks_the_visually_closer_candidate(self):
        anchor = disc(40)
        candidates = [disc(40, fill=BLUE),                  # right shape, wrong hue
                      disc(42),                             # the match
                      disc(20, fill=(0, 200, 60, 255))]     # wrong both ways
        self.assertEqual(consistency.nearest(candidates, anchor), 1)

    def test_accepts_precomputed_features(self):
        anchor = consistency.features(disc(40))
        candidates = [consistency.features(disc(40, fill=BLUE)),
                      consistency.features(disc(42))]
        self.assertEqual(consistency.nearest(candidates, anchor), 1)

    def test_a_blank_candidate_never_wins(self):
        blank = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
        self.assertEqual(consistency.nearest([blank, disc(41)], disc(40)), 1)
        with self.assertRaises(ValueError):
            consistency.nearest([blank], disc(40))

    def test_distance_is_zero_against_itself(self):
        feats = consistency.features(disc(40))
        self.assertEqual(consistency.distance(feats, feats), 0.0)


class SnapPalette(unittest.TestCase):

    def setUp(self):
        self.im = Image.new("RGBA", (4, 1))
        self.im.putpixel((0, 0), (233, 131, 55, 255))    # near #FF7A28
        self.im.putpixel((1, 0), (250, 240, 215, 128))   # near #FFF5DC, part alpha
        self.im.putpixel((2, 0), (40, 40, 44, 255))      # near nothing declared
        self.im.putpixel((3, 0), (120, 60, 20, 0))       # transparent

    def test_every_pixel_lands_on_a_declared_colour(self):
        out = consistency.snap_palette(self.im, PALETTE)
        declared = set(consistency._hex_rgb(h) for h in PALETTE)
        for x in range(4):
            self.assertIn(out.getpixel((x, 0))[:3], declared)
        self.assertEqual(out.getpixel((0, 0))[:3], (255, 122, 40))
        self.assertEqual(out.getpixel((1, 0))[:3], (255, 245, 220))

    def test_a_dark_pixel_does_not_become_black(self):
        """Padding the unused palette slots with zeros makes black a real entry,
        so (40,40,44) snaps to (0,0,0) -- a colour nobody declared."""
        out = consistency.snap_palette(self.im, PALETTE)
        self.assertNotEqual(out.getpixel((2, 0))[:3], (0, 0, 0))

    def test_alpha_is_reattached_unchanged(self):
        out = consistency.snap_palette(self.im, PALETTE)
        self.assertEqual([out.getpixel((x, 0))[3] for x in range(4)],
                         [255, 128, 255, 0])

    def test_a_bad_palette_entry_is_rejected(self):
        for bad in (["#GGGGGG"], ["12345"], ["#FF7A28", "orange"]):
            with self.assertRaises(ValueError):
                consistency.snap_palette(self.im, bad)
        with self.assertRaises(ValueError):
            consistency.snap_palette(self.im, [])


if __name__ == "__main__":
    unittest.main()
