# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
"""Rebuild the gradient tables of the phantom protocols from their published sources.

Each table is written to ``protocols/<name>.b``, one ``x y z b`` row per volume:
unit vectors in world (RAS) coordinates, as prescribed, and b-values in s/mm².
Sources come in different frames, converted here:

- ``hcpya``: a Siemens diffusion vector set, in device coordinates (which are
  LPS for a head-first supine subject), with the b-value in the vector's norm.
- ``hbn``, ``aomic``: FSL b-vectors, in the voxel frame of the image (whose
  header is read to convert them).
- ``abcd``: FSL b-vectors without an image; the grid is taken to be LAS, as
  ``dcm2niix`` writes the axial Siemens images of the same sites.
- ``sherbrooke``, ``stanford``: b-vectors in the voxel frame of an image whose
  axes are those of RAS, as DIPY reads them.

Diffusion contrast does not depend on the sign of a b-vector, so these frames
only matter for the sign of simulated eddy-current distortions.

Usage::

    python tools/phantoms/fetch_protocols.py [--check]
"""

import argparse
import hashlib
import io
import re
import sys
import urllib.request
import zlib
from pathlib import Path

import nibabel as nb
import numpy as np

HERE = Path(__file__).parent

SOURCES = {
    "hcpya": {"dvs": "https://wiki.humanconnectome.org/docs/assets/HCP_Diffusion_Vectors.txt"},
    "hbn": {
        "stem": "https://fcp-indi.s3.amazonaws.com/data/Projects/HBN/MRI/Site-CBIC/"
        "sub-NDARAA396TWZ/dwi/sub-NDARAA396TWZ_acq-64dir_dwi",
    },
    "abcd": {
        "bvec": "https://raw.githubusercontent.com/DCAN-Labs/abcd-dicom2bids/"
        "aee6ca8e5d473dab86cd142681acb6752a7945c6/src/ABCD_Release_2.0_Diffusion_Tables/"
        "Siemens_bvecs.txt",
        "bval": "https://raw.githubusercontent.com/DCAN-Labs/abcd-dicom2bids/"
        "aee6ca8e5d473dab86cd142681acb6752a7945c6/src/ABCD_Release_2.0_Diffusion_Tables/"
        "Siemens_bvals.txt",
    },
    "sherbrooke": {
        "bvec": "https://workshop.dipy.org/services/data/researchworks/bitstream/handle/"
        "1773/38475/HARDI193.bvec",
        "bval": "https://workshop.dipy.org/services/data/researchworks/bitstream/handle/"
        "1773/38475/HARDI193.bval",
    },
    "stanford": {
        "bvec": "https://stacks.stanford.edu/file/druid:yx282xq2090/dwi.bvecs",
        "bval": "https://stacks.stanford.edu/file/druid:yx282xq2090/dwi.bvals",
    },
    "aomic": {"stem": "https://s3.amazonaws.com/openneuro.org/ds002790/sub-0001/dwi/sub-0001_dwi"},
}
"""Where each protocol's table is published."""

DVS_VECTOR = re.compile(r"Vector\[\d+\]\s*=\s*\(\s*(\S+),\s*(\S+),\s*(\S+)\s*\)")
"""One vector of a Siemens diffusion vector set."""

LPS_TO_RAS = np.array([-1.0, -1.0, 1.0])
LAS_TO_RAS = np.array([-1.0, 1.0, 1.0])


def _get(url: str, first_bytes: int | None = None) -> bytes:
    request = urllib.request.Request(url)
    if first_bytes is not None:
        request.add_header("Range", f"bytes=0-{first_bytes - 1}")
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def _nifti_affine(url: str) -> np.ndarray:
    """The affine of a remote ``.nii.gz``, from its first bytes only."""
    header = zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(_get(url, 65536))[:352]
    return nb.Nifti1Header.from_fileobj(io.BytesIO(header)).get_best_affine()


def _rasb(vectors: np.ndarray, bvals: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit = np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 1e-6)
    unit[bvals < 50] = 0.0
    return np.column_stack((unit, bvals))


def from_dvs(text: str) -> np.ndarray:
    """A Siemens diffusion vector set with the b-value in the norm (b-max 3000)."""
    vectors = np.array(
        [[float(value) for value in match.groups()] for match in DVS_VECTOR.finditer(text)]
    )
    # Vectors are stored to three decimals: the norm gives b within 4 s/mm² of its shell
    bvals = np.round(3000.0 * (vectors**2).sum(axis=1), -3)
    return _rasb(vectors * LPS_TO_RAS, bvals)


def from_fsl(vectors: np.ndarray, bval: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """FSL b-vectors, ``(n, 3)``, in the voxel frame of an image with ``affine``."""
    rotation = affine[:3, :3] / np.linalg.norm(affine[:3, :3], axis=0)
    vectors = vectors.copy()
    if np.linalg.det(rotation) > 0:
        # FSL's voxel frame flips the first axis of a neurological (RAS-like) grid
        vectors[:, 0] *= -1.0
    return _rasb(vectors @ rotation.T, bval)


def build(name: str) -> tuple[np.ndarray, dict[str, str]]:
    """Return a protocol's table and the sha256 of each source file."""
    source = SOURCES[name]
    texts = {}
    if "dvs" in source:
        texts["dvs"] = _get(source["dvs"])
        table = from_dvs(texts["dvs"].decode())
    else:
        stem = source.get("stem")
        urls = {key: source[key] if stem is None else f"{stem}.{key}" for key in ("bvec", "bval")}
        texts = {key: _get(url) for key, url in urls.items()}
        bvec = np.loadtxt(io.BytesIO(texts["bvec"]))
        bval = np.loadtxt(io.BytesIO(texts["bval"]))
        vectors = bvec.T if bvec.shape[0] == 3 else bvec
        if stem is not None:
            table = from_fsl(vectors, bval, _nifti_affine(f"{stem}.nii.gz"))
        elif name == "abcd":
            table = from_fsl(vectors, bval, np.diag([*LAS_TO_RAS, 1.0]))
        else:
            # DIPY reads these in the voxel frame of an image whose axes are those of RAS
            table = _rasb(vectors, bval)
    return table, {key: hashlib.sha256(text).hexdigest() for key, text in texts.items()}


def render(name: str, table: np.ndarray) -> str:
    lines = [
        f"# {name}: one row per volume, x y z (unit vector, world RAS) and b (s/mm2)",
        "# written by tools/phantoms/fetch_protocols.py",
    ]
    lines += [f"{x: .6f} {y: .6f} {z: .6f} {b:.0f}" for x, y, z, b in table]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="compare, do not write")
    parser.add_argument("protocols", nargs="*", default=list(SOURCES))
    args = parser.parse_args(argv)
    status = 0
    for name in args.protocols:
        table, sources = build(name)
        text, target = render(name, table), HERE / "protocols" / f"{name}.b"
        digest = hashlib.sha256(text.encode()).hexdigest()
        print(f"{name}: {len(table)} volumes, sha256 {digest}, sources {sources}")
        if args.check:
            if not target.exists() or target.read_text() != text:
                print(f"  {target} differs from its sources")
                status = 1
        else:
            target.parent.mkdir(exist_ok=True)
            target.write_text(text)
    return status


if __name__ == "__main__":
    sys.exit(main())
