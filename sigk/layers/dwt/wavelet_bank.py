from sigk.layers.dwt.wavelet import Wavelet


CDF_9_7_ANALYSIS_LOW_FILTER_BANK: list[float] = [
    0,  # extension
    0.026748757410810,  # -4
    -0.016864118442875,  # -3
    -0.078223266528990,  # -2
    0.266864118442875,  # -1
    0.602949018236360,  # 0
    0.266864118442875,  # 1
    -0.078223266528990,  # 2
    -0.016864118442875,  # 3
    0.026748757410810,  # 4
]

CDF_9_7_ANALYSIS_HIGH_FILTER_BANK: list[float] = [
    0,  # -4
    0.091271763114250,  # -3
    -0.057543526228500,  # -2
    -0.591271763114250,  # -1
    1.115087052457000,  # 0
    -0.591271763114250,  # 1
    -0.057543526228500,  # 2
    0.091271763114250,  # 3
    0,  # 4
    0,  # extension
]

CDF_9_7_SYNTHESIS_LOW_FILTER_BANK: list[float] = [
    0,  # -4
    -0.091271763114250,  # -3
    -0.057543526228500,  # -2
    0.591271763114250,  # -1
    1.115087052457000,  # 0
    0.591271763114250,  # 1
    -0.057543526228500,  # 2
    -0.091271763114250,  # 3
    0,  # 4
    0,  # extension
]

CDF_9_7_SYNTHESIS_HIGH_FILTER_BANK: list[float] = [
    0,  # extension
    0.026748757410810,  # -4
    0.016864118442875,  # -3
    -0.078223266528990,  # -2
    -0.266864118442875,  # -1
    0.602949018236360,  # 0
    -0.266864118442875,  # 1
    -0.078223266528990,  # 2
    0.016864118442875,  # 3
    0.026748757410810,  # 4
]


COHEN_DAUBECHIES_FEAUVEAU_9_7_WAVELET = Wavelet(
    "cdf 9/7",
    CDF_9_7_ANALYSIS_LOW_FILTER_BANK,
    CDF_9_7_ANALYSIS_HIGH_FILTER_BANK,
    CDF_9_7_SYNTHESIS_LOW_FILTER_BANK,
    CDF_9_7_SYNTHESIS_HIGH_FILTER_BANK,
)

LGT_5_3_ANALYSIS_LOW_FILTER_BANK: list[float] = [
    0,  # extension
    -0.125,  # -2
    0.25,  # -1
    0.75,  # 0
    0.25,  # 1
    -0.125,  # 2
]

LGT_5_3_ANALYSIS_HIGH_FILTER_BANK: list[float] = [
    0,  # -2
    -0.5,  # -1
    1,  # 0
    -0.5,  # 1
    0,  # 2
    0,  # extension
]

LGT_5_3_SYNTHESIS_LOW_FILTER_BANK: list[float] = [
    0,  # -2
    0.5,  # -1
    1,  # 0
    0.5,  # 1
    0,  # 2
    0,  # extension
]

LGT_5_3_SYNTHESIS_HIGH_FILTER_BANK: list[float] = [
    0,  # extension
    -0.125,  # -2
    -0.25,  # -1
    0.75,  # 0
    -0.25,  # 1
    -0.125,  # 2
]


LE_GALL_TABATABAI_5_3_WAVELET = Wavelet(
    "lgt 5/3",
    LGT_5_3_ANALYSIS_LOW_FILTER_BANK,
    LGT_5_3_ANALYSIS_HIGH_FILTER_BANK,
    LGT_5_3_SYNTHESIS_LOW_FILTER_BANK,
    LGT_5_3_SYNTHESIS_HIGH_FILTER_BANK,
)
