import pytest

from emotion_render import Refused
from emotion_render.engines import ENGINES, choose_device

TURBO = ENGINES["turbo"]
MULTILINGUAL = ENGINES["multilingual"]
GTX_1650_MB = 3717  # 3.63 GiB usable, as torch reported it on eury (A4, 2026-11-23)

# Peak RSS measured on eury in A4, in MB.
MEASURED_PEAK_RSS = {
    ("turbo", "cpu"): 6427,
    ("turbo", "cuda"): 4854,
    ("multilingual", "cpu"): 6938,
}


@pytest.mark.parametrize("key, peak", MEASURED_PEAK_RSS.items())
def test_the_ram_guard_sits_above_every_measured_peak(key, peak):
    engine, device = key
    assert ENGINES[engine].min_ram_mb[device] > peak


def test_turbo_fits_the_measured_vram_peak_on_the_gtx_1650():
    assert 3350 < TURBO.min_vram_mb <= GTX_1650_MB


def test_auto_takes_the_gpu_for_turbo():
    assert choose_device("auto", TURBO, GTX_1650_MB) == "cuda"


def test_auto_keeps_multilingual_on_cpu():
    assert choose_device("auto", MULTILINGUAL, GTX_1650_MB) == "cpu"


def test_auto_without_a_gpu_is_cpu():
    assert choose_device("auto", TURBO, None) == "cpu"


def test_cpu_is_always_cpu():
    assert choose_device("cpu", MULTILINGUAL, GTX_1650_MB) == "cpu"


def test_cuda_for_multilingual_is_refused():
    with pytest.raises(Refused) as e:
        choose_device("cuda", MULTILINGUAL, GTX_1650_MB)
    assert "--device cpu" in str(e.value)


def test_cuda_on_a_card_too_small_is_refused():
    with pytest.raises(Refused):
        choose_device("cuda", TURBO, 2048)


def test_cuda_without_a_gpu_is_refused():
    with pytest.raises(Refused):
        choose_device("cuda", TURBO, None)
