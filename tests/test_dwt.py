from sigk.layers.dwt import (
    COHEN_DAUBECHIES_FEAUVEAU_9_7_WAVELET,
    LE_GALL_TABATABAI_5_3_WAVELET,
    Wavelet,
    Dwt2D,
    IDwt2D,
)

from torch import Tensor, abs as tensor_abs, rand, float64, from_numpy

from pytest import mark

from pywt import dwt2

from itertools import chain
from typing import Sequence, Any


EPSILON: float = 1e-10


def equals(left: Tensor, right: Tensor) -> bool:
    return tensor_abs((left - right)).max() < EPSILON


TEST_WAVELETS: Wavelet = [
    COHEN_DAUBECHIES_FEAUVEAU_9_7_WAVELET,
    LE_GALL_TABATABAI_5_3_WAVELET,
    Wavelet("db1"),
    Wavelet("bior4.4"),
    Wavelet("bior5.5"),
    Wavelet("db5"),
    Wavelet("coif2"),
    Wavelet("sym5"),
    Wavelet("rbio3.7"),
]


def for_all_test_wavelets(
    *args: Sequence[int] | str,
) -> list[tuple[Sequence[int] | str | Wavelet]]:
    return [(*args, wavelet) for wavelet in TEST_WAVELETS]


def concatentate_args(*args: list[Any]) -> list[Any]:
    return list(chain(*args))


DWT1D_TEST_CASES: list[tuple[tuple[int, int, int], str, str, Wavelet]] = (
    concatentate_args(
        for_all_test_wavelets((256, 64, 128), "zeros", "zero"),
        for_all_test_wavelets((256, 64, 128), "reflect", "reflect"),
        for_all_test_wavelets((256, 64, 128), "replicate", "constant"),
        for_all_test_wavelets((256, 64, 128), "circular", "periodic"),
    )
)

DWT2D_TEST_CASES: list[tuple[tuple[int, int, int], str, str, Wavelet]] = (
    concatentate_args(
        for_all_test_wavelets((128, 16, 128, 128), "zeros", "zero"),
        for_all_test_wavelets((128, 16, 128, 128), "reflect", "reflect"),
        for_all_test_wavelets((128, 16, 128, 128), "replicate", "constant"),
        for_all_test_wavelets((128, 16, 128, 128), "circular", "periodic"),
    )
)


@mark.parametrize(
    "shape, dwt_pad, test_pad, wavelet",
    DWT2D_TEST_CASES,
)
def test_dwt2d_separate(shape, dwt_pad, test_pad, wavelet):
    sequence = rand(*shape, dtype=float64)
    _dwt = Dwt2D(shape[1], wavelet=wavelet, padding_mode=dwt_pad, dtype=float64)
    ll, (lh, hl, hh) = dwt2(sequence.numpy(), wavelet=wavelet.as_pywt, mode=test_pad)
    ll, lh, hl, hh = from_numpy(ll), from_numpy(lh), from_numpy(hl), from_numpy(hh)
    ll_, lh_, hl_, hh_ = _dwt(sequence)
    assert ll.shape == ll_.shape
    assert lh.shape == lh_.shape
    assert hl.shape == hl_.shape
    assert hh.shape == hh_.shape
    assert equals(ll, ll_)
    assert equals(lh, lh_)
    assert equals(hl, hl_)
    assert equals(hh, hh_)


@mark.parametrize(
    "shape, dwt_pad, test_pad, wavelet",
    DWT2D_TEST_CASES,
)
def test_dwt2d_none(shape, dwt_pad, test_pad, wavelet):
    sequence = rand(*shape, dtype=float64)
    _dwt = Dwt2D(shape[1], wavelet=wavelet, padding_mode=dwt_pad, dtype=float64)
    ll, (lh, hl, hh) = dwt2(sequence.numpy(), wavelet=wavelet.as_pywt, mode=test_pad)
    ll, lh, hl, hh = from_numpy(ll), from_numpy(lh), from_numpy(hl), from_numpy(hh)
    result = _dwt(sequence, "none")
    assert result.shape[-1] == ll.shape[-1]
    assert result.shape[-2] == ll.shape[-2]
    assert result.shape[-3] == ll.shape[-3] * 4
    ll_ = result[..., 3::4, :, :]
    lh_ = result[..., 1::4, :, :]
    hl_ = result[..., 2::4, :, :]
    hh_ = result[..., ::4, :, :]
    assert ll.shape == ll_.shape
    assert lh.shape == lh_.shape
    assert hl.shape == hl_.shape
    assert hh.shape == hh_.shape
    assert equals(ll, ll_)
    assert equals(lh, lh_)
    assert equals(hl, hl_)
    assert equals(hh, hh_)


@mark.parametrize(
    "shape, dwt_pad, test_pad, wavelet",
    DWT2D_TEST_CASES,
)
def test_dwt2d_dimension(shape, dwt_pad, test_pad, wavelet):
    sequence = rand(*shape, dtype=float64)
    _dwt = Dwt2D(shape[1], wavelet=wavelet, padding_mode=dwt_pad, dtype=float64)
    ll, (lh, hl, hh) = dwt2(sequence.numpy(), wavelet=wavelet.as_pywt, mode=test_pad)
    ll, lh, hl, hh = from_numpy(ll), from_numpy(lh), from_numpy(hl), from_numpy(hh)
    result = _dwt(sequence, "dimension")
    assert result.dim() == ll.dim() + 1
    ll_, lh_, hl_, hh_ = result.chunk(4, dim=2)
    ll_ = ll_.squeeze()
    lh_ = lh_.squeeze()
    hl_ = hl_.squeeze()
    hh_ = hh_.squeeze()
    assert ll_.shape == ll.shape
    assert lh_.shape == lh.shape
    assert hl_.shape == hl.shape
    assert hh_.shape == hh.shape
    assert equals(ll, ll_)
    assert equals(lh, lh_)
    assert equals(hl, hl_)
    assert equals(hh, hh_)


@mark.parametrize(
    "shape, dwt_pad, splitting_mode, wavelet",
    concatentate_args(
        for_all_test_wavelets((128, 16, 128, 128), "zeros", "none"),
        for_all_test_wavelets((128, 16, 128, 128), "zeros", "separate"),
        for_all_test_wavelets((128, 16, 128, 128), "zeros", "dimension"),
        for_all_test_wavelets((128, 16, 128, 128), "reflect", "none"),
        for_all_test_wavelets((128, 16, 128, 128), "reflect", "separate"),
        for_all_test_wavelets((128, 16, 128, 128), "reflect", "dimension"),
        for_all_test_wavelets((128, 16, 128, 128), "replicate", "none"),
        for_all_test_wavelets((128, 16, 128, 128), "replicate", "separate"),
        for_all_test_wavelets((128, 16, 128, 128), "replicate", "dimension"),
        for_all_test_wavelets((128, 16, 128, 128), "circular", "none"),
        for_all_test_wavelets((128, 16, 128, 128), "circular", "separate"),
        for_all_test_wavelets((128, 16, 128, 128), "circular", "dimension"),
    ),
)
def test_idwt2d(shape, dwt_pad, splitting_mode, wavelet):
    sequence = rand(*shape, dtype=float64)
    _dwt = Dwt2D(shape[1], wavelet=wavelet, dtype=float64, padding_mode=dwt_pad)
    result = _dwt(sequence, splitting_mode)
    _idwt = IDwt2D(shape[1], wavelet=wavelet, dtype=float64)
    recon = _idwt(result, splitting_mode)
    assert recon.shape == sequence.shape
    assert equals(recon, sequence)
