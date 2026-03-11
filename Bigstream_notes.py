"""BigStream registration notes (spatial transcriptomics)

-------------------------------------------------------------------------------
PATHS / FILE NAMING (cluster + local)
-------------------------------------------------------------------------------

RAW files (used for masks + global alignment):

DEFAULT_FIXED_NAME = "exp_001_fish2_s07_pre_GCaMP_cropped.tif"
DEFAULT_MOVING_NAME = (
    "2025-10-13_16-04-47_fish002_setup1_arena0_MW_preprocessed_data_repeat00_tile000_950nm_0_flippedxz_CARE.tif"
)

CONVOLUTED files (used for piecewise only):

DEFAULT_FIXED_NAME_CONVOLUTED = "exp_001_fish2_s07_pre_GCaMP_gf50.tif"
DEFAULT_MOVING_NAME_CONVOLUTED = (
    "2025-10-13_16-04-47_fish002_setup1_arena0_MW_preprocessed_data_repeat00_tile000_950nm_0_flippedxz_CARE_gf40.tif"
)

-------------------------------------------------------------------------------
CURRENT “BEST” PIECEWISE CONFIG (NO DEFORM)
-------------------------------------------------------------------------------

# Best so far: multi-start random (MMI) followed by rigid refine (ANC)
# Large blocks capture global drift; overlap helps continuity.
# exp_001_fish2_20260228_133104 """

PIECEWISE_STEPS_LARGE_BEST = [
    ("random", dict(
        random_iterations=5000,
        nreturn=3,
        max_translation=22.0,
        max_rotation=0.07,
        max_scale=None,
        max_shear=None,
        alignment_spacing=5.0,
        metric="MMI",
        metric_args={"numberOfHistogramBins": 20},
        sampling="RANDOM",
        sampling_percentage=0.25,
        shrink_factors=(2, 1),
        smooth_sigmas=(1.5, 0.0),
    )),

    ("rigid", dict(
        metric="ANC",
        metric_args={"radius": 10},
        optimizer="RSGD",
        optimizer_args={
            "learningRate": 0.05,
            "minStep": 2e-7,
            "numberOfIterations": 400,
        },
        sampling="RANDOM",
        sampling_percentage=0.25,
        shrink_factors=(2, 1),
        smooth_sigmas=(1.5, 0.0),
    )),
]

BLOCKSIZE_LARGE = (320, 320, 320)
OVERLAP_LARGE = 0.35
"""

-------------------------------------------------------------------------------
ADDING LARGER SCALE + SHEAR FREEDOM IN THE RANDOM INITIALIZER + SMALL DEFORM
-------------------------------------------------------------------------------

Increasing the random+adding an affine step did not help dramatically. Rather some lateral disotrtion appeared. 
exp_001_fish2_20260302_132001

"""
PIECEWISE_6000 = [
    ("random", dict(
        random_iterations=6000, nreturn=3,
        max_translation=24.0, max_rotation=0.075,
        max_scale=None, max_shear=None,
        alignment_spacing=5.0,
        metric="MMI", metric_args={"numberOfHistogramBins": 20},
        sampling="RANDOM", sampling_percentage=0.3,
        shrink_factors=(2, 1), smooth_sigmas=(1.5, 0.0),
    )),
    ("rigid", dict(
        metric="ANC", metric_args={"radius": 12},
        optimizer="RSGD",
        optimizer_args=dict(learningRate=0.05, minStep=2e-7, numberOfIterations=450),
        sampling="RANDOM", sampling_percentage=0.3,
        shrink_factors=(2, 1), smooth_sigmas=(1.5, 0.0),
        initial_condition="IDENTITY",
    )),
    ("affine", dict(
        metric="ANC", metric_args={"radius": 10},
        optimizer="RSGD",
        optimizer_args=dict(learningRate=0.02, minStep=5e-7, numberOfIterations=220),
        sampling="RANDOM", sampling_percentage=0.25,
        shrink_factors=(4, 2, 1), smooth_sigmas=(2.0, 1.0, 0.0),
        initial_condition="IDENTITY",
    )),
]

BLOCKSIZE_6000 = (320, 320, 320)
OVERLAP_6000 = 0.35

"""
Bumping random_iterations, max_translation and max_rotation. 
No clear improvement, rather the hindbrain edges got worse. 
exp_001_fish2_20260228_133105

"""
PIECEWISE_9000 = [
    ("random", dict(random_iterations=9000, nreturn=3,
        max_translation=24.0, max_rotation=0.075,
        max_scale=None, max_shear=None,
        alignment_spacing=5.0,
        metric="MMI", metric_args={"numberOfHistogramBins": 20},
        sampling="RANDOM", sampling_percentage=0.25,
        shrink_factors=(2, 1), smooth_sigmas=(1.8, 0.0),
    )),
    ("rigid", dict(
        metric="ANC", metric_args={"radius": 10},
        optimizer="RSGD",
        optimizer_args=dict(learningRate=0.045, minStep=2e-7, numberOfIterations=450),
        sampling="RANDOM", sampling_percentage=0.25,
        shrink_factors=(2, 1), smooth_sigmas=(1.5, 0.0),
    )),
]
BLOCKSIZE_9000 = (320, 320, 320)
OVERLAP_9000 = 0.35

"""

-------------------------------------------------------------------------------
ADDING DEFORMATION STEP 
-------------------------------------------------------------------------------

Increasing deformation capabilities made things worse, not better:
exp_001_fish2_20260301_133104

"""
PIECEWISE_5000_WITH_DEFORM = [
    ("random", dict(
        random_iterations=5000, nreturn=3,
        max_translation=22.0, max_rotation=0.07,
        max_scale=None, max_shear=None,
        alignment_spacing=5.0,
        metric="MMI", metric_args={"numberOfHistogramBins": 20},
        sampling="RANDOM", sampling_percentage=0.25,
        shrink_factors=(2, 1), smooth_sigmas=(1.5, 0.0),
    )),
    ("rigid", dict(
        metric="ANC", metric_args={"radius": 10},
        optimizer="RSGD",
        optimizer_args=dict(learningRate=0.05, minStep=5e-7, numberOfIterations=400),
        sampling="RANDOM", sampling_percentage=0.25,
        shrink_factors=(2, 1), smooth_sigmas=(1.5, 0.0),
    )),
    ("deform", dict(
        control_point_spacing=300.0, control_point_levels=[1],
        metric="MMI", metric_args={"numberOfHistogramBins": 24},
        optimizer="RSGD",
        optimizer_args=dict(learningRate=0.02, minStep=5e-6, numberOfIterations=50),
        sampling="RANDOM", sampling_percentage=0.1,
        shrink_factors=(2,), smooth_sigmas=(2.0,),
    )),
]

BLOCKSIZE_5000_WITH_DEFORM = (320, 320, 320)
OVERLAP_5000_WITH_DEFORM = 0.35
"""

Increasing random_iterations, max_translation and max_rotation and making 2 rigid steps
No clear improvement. 
exp_001_fish2_20260302_212952
exp_001_fish2_20260303_084409

"""
PIECEWISE_9000_FULL = [
    ("random", dict(
        random_iterations=9000, nreturn=5,
        max_translation=26.0, max_rotation=0.085,
        max_scale=None, max_shear=None,
        alignment_spacing=6.0,
        metric="MMI", metric_args={"numberOfHistogramBins": 16},
        sampling="RANDOM", sampling_percentage=0.4,
        shrink_factors=(2, 1), smooth_sigmas=(1.8, 0.0),
    )),
    ("rigid", dict(
        metric="ANC", metric_args={"radius": 10},
        optimizer="RSGD",
        optimizer_args=dict(learningRate=0.06, minStep=5e-7, numberOfIterations=500),
        sampling="RANDOM", sampling_percentage=0.35,
        shrink_factors=(2, 1), smooth_sigmas=(1.5, 0.0),
        initial_condition="IDENTITY",
    )),
    ("rigid", dict(
        metric="ANC", metric_args={"radius": 6},
        optimizer="RSGD",
        optimizer_args=dict(learningRate=0.035, minStep=5e-7, numberOfIterations=300),
        sampling="RANDOM", sampling_percentage=0.35,
        shrink_factors=(2, 1), smooth_sigmas=(1.0, 0.0),
        initial_condition="IDENTITY",
    )),
    ("deform", dict(
        control_point_spacing=200.0, control_point_levels=[1],
        metric="ANC", metric_args={"radius": 8},
        optimizer="RSGD",
        optimizer_args=dict(learningRate=0.015, minStep=2e-5, numberOfIterations=40),
        sampling="RANDOM", sampling_percentage=0.15,
        shrink_factors=(2,), smooth_sigmas=(3.0,),
    )),
]

BLOCKSIZE_9000_FULL = (320, 320, 320)
OVERLAP_9000_FULL = 0.35
"""

Reducing the deform step control spacing. Quality got worse at the level of the midline. 
exp_001_fish2_20260303_190715

"""
PIECEWISE_5000_DEFORM = [
    ("random", dict(
        random_iterations=5000, nreturn=5,
        max_translation=22.0, max_rotation=0.07,
        max_scale=None, max_shear=None,
        alignment_spacing=6.0,
        metric="MMI", metric_args={"numberOfHistogramBins": 16},
        sampling="RANDOM", sampling_percentage=0.4,
        shrink_factors=(2, 1), smooth_sigmas=(1.8, 0.0),
    )),
    ("rigid", dict(
        metric="ANC", metric_args={"radius": 10},
        optimizer="RSGD",
        optimizer_args=dict(learningRate=0.05, minStep=2e-7, numberOfIterations=400),
        sampling="RANDOM", sampling_percentage=0.25,
        shrink_factors=(2, 1), smooth_sigmas=(1.5, 0.0),
    )),
    ("deform", dict(
        control_point_spacing=140.0, control_point_levels=[1],
        metric="ANC", metric_args={"radius": 8},
        optimizer="RSGD",
        optimizer_args=dict(learningRate=0.01, minStep=1e-6, numberOfIterations=50),
        sampling="RANDOM", sampling_percentage=0.1,
        shrink_factors=(2,), smooth_sigmas=(2.5,),
    )),
]

BLOCKSIZE_5000_DEFORM = (320, 320, 320)
OVERLAP_5000_DEFORM = 0.35

"""
-------------------------------------------------------------------------------
INCREASING AFFINE FREEDOM (SCALE + SHEAR) IN RANDOM INIT
-------------------------------------------------------------------------------

# Exp1 
# Attempted more aggressive local flexibility with thin-Z blocks and
# larger scale freedom, to specifically target tectal neuropil mismatch.
# This actually made things a bit better especially at the edges of the Hb. 
# This is the best so far, betee than: exp_001_fish2_20260228_133104 
exp_001_fish2_20260303_205121

"""
PIECEWISE_6000_AFFINE = [
    ("random", dict(
        random_iterations=6000, nreturn=5,
        max_translation=22.0, max_rotation=0.07,
        max_scale=1.008, max_shear=0.01,
        alignment_spacing=6.0,
        metric="MMI", metric_args={"numberOfHistogramBins": 16},
        sampling="RANDOM", sampling_percentage=0.4,
        shrink_factors=(2, 1), smooth_sigmas=(1.8, 0.0),
    )),
    ("affine", dict(
        metric="MMI", metric_args={"numberOfHistogramBins": 16},
        optimizer="RSGD",
        optimizer_args=dict(learningRate=0.02, minStep=5e-7, numberOfIterations=200),
        sampling="RANDOM", sampling_percentage=0.25,
        shrink_factors=(4, 2, 1), smooth_sigmas=(2.0, 1.0, 0.0),
        initial_condition="IDENTITY",
    )),
]

BLOCKSIZE_6000_AFFINE = (320, 320, 320)
OVERLAP_6000_AFFINE = 0.35
"""

-------------------------------------------------------------------------------
REDUCING THE BLOCK SIZE TO 160
-------------------------------------------------------------------------------

Zig-Zag patterns. 
exp_001_fish2_20260304_093921. 

"""

PIECEWISE_STEPS_LARGE_THINZ_FAILED = [
    ("random", dict(
        random_iterations=6000,
        nreturn=5,
        max_translation=22.0,
        max_rotation=0.07,
        max_scale=1.05,          # ~±5% (much looser than earlier ±0.8%)
        max_shear=0.010,
        alignment_spacing=3.0,
        metric="MMI",
        metric_args={"numberOfHistogramBins": 16},
        sampling="NONE",
        shrink_factors=(2, 1),
        smooth_sigmas=(1.8, 0.0),
    )),

    ("affine", dict(
        metric="ANC",
        metric_args={"radius": 12},
        optimizer="RSGD",
        optimizer_args=dict(
            learningRate=0.02,
            minStep=5e-7,
            numberOfIterations=500,
        ),
        sampling="NONE",
        shrink_factors=(2, 1),
        smooth_sigmas=(1.0, 0.0),
    )),
]

BLOCKSIZE_THINZ_FAILED = (160, 160, 16)
OVERLAP_THINZ_FAILED = 0.50

"""
Outcome:
 - Did NOT correct tectal neuropil misalignment.
 - Produced visible zig-zag / chevron artifacts.
 - Likely caused by:
     * thin Z blocks (16 slices) leading to inter-slice inconsistencies,
     * overly permissive scale (±5%),
     * sampling="NONE" increasing sensitivity to local intensity structure.
 - Conclusion: increasing local affine freedom is not the right lever here.
"""

"""
# Exp2 
Attempting to remove the zig-zag by removing the scaling and shearing. Also replace the affine by rigid. 
Also testing different bloc sizes : blocksize0 = (19, 160, 160) and blocksize0 = (160, 160, 160). 
Problem: block size gaps in the moving volume. 
Experiments tags: 
exp_001_fish2_20260304_112834
exp_001_fish2_20260304_134811

"""
PIECEWISE_STEPS_LARGE = [
    ("random", dict(
        random_iterations=6000, nreturn=5,
        max_translation=22.0, max_rotation=0.07,
        max_scale=None, max_shear=None,
        alignment_spacing=3.0,
        metric="MMI", metric_args={"numberOfHistogramBins": 16},
        sampling="RANDOM", sampling_percentage=0.30,
        shrink_factors=(2, 1), smooth_sigmas=(1.8, 0.0),
    )),
    ("rigid", dict(
        metric="ANC", metric_args={"radius": 12},
        optimizer="RSGD",
        optimizer_args=dict(learningRate=0.02, minStep=5e-7, numberOfIterations=500),
        sampling="RANDOM", sampling_percentage=0.25,
        shrink_factors=(2, 1), smooth_sigmas=(1.0, 0.0),
    )),
]

blocksize0 = (160, 160, 160)
overlap0  = 0.50   # try 0.40 if you see seams / block inconsistencies

"""
Outcome:
 - Zig-zag pattern is gone, but local misalignments remain, especially in the neuropil.
 - Either block sizes produce the same effects: block size gaps in the moving volume. 
 - Registration got a bit better on the PVZ edges compared to smaller blocks with more scale/shear freedom. 

-------------------------------------------------------------------------------
ALIGNING THE WHOLE FIX STACK IN A SINGLE CALL
-------------------------------------------------------------------------------

Base : This call works very well for these volumes: 
MOVING_PATH_CONVOLUTED = BASE / "data/exp1_110425/2p_stacks/2025-10-13_16-04-47_fish002_setup1_arena0_MW_preprocessed_data_repeat00_tile000_950nm_0_flippedxz_CARE_gf40.tif"
FIXED_PATH_CONVOLUTED  = BASE / "data/exp1_110425/oct_confocal_stacks/fish2/prealigned/exp_001_fish2_s05-s09_montaged_MattesMI_GCaMP_gf50.tif"
again the non-coonvoluted volumes are use for the maskes. 

"""
("rigid", dict(
    metric="MMI", metric_args={"numberOfHistogramBins": 32},
    optimizer="RSGD",
    optimizer_args=dict(learningRate=1.0, minStep=1e-4, numberOfIterations=600),
    sampling="RANDOM", sampling_percentage=0.20,
    shrink_factors=(32, 16, 8, 4, 2, 1),
    smooth_sigmas=(24.0, 12.0, 6.0, 3.0, 1.0, 0.0),
    initial_condition="CENTER",
)),
("random", dict(
    random_iterations=8000, nreturn=5,
    max_translation=20.0, max_rotation=0.25,
    max_scale=1.03, max_shear=0.03,
    alignment_spacing=8.0,
    metric="MMI", metric_args={"numberOfHistogramBins": 24},
    sampling="RANDOM", sampling_percentage=0.10,
    shrink_factors=(16, 8, 4),
    smooth_sigmas=(6.0, 3.0, 1.5),
)),
("affine", dict(
    metric="MMI", metric_args={"numberOfHistogramBins": 48},
    optimizer="RSGD",
    optimizer_args=dict(learningRate=0.12, minStep=5e-7, numberOfIterations=1200),
    sampling="RANDOM", sampling_percentage=0.20,
    shrink_factors=(16, 8, 4, 2, 1),
    smooth_sigmas=(6.0, 3.0, 1.5, 0.5, 0.0),
)),

"""
Outcome: very good intitialization for fish2. Edges are not snapping everywhere but good enough for regional refinement. 
Proceed to next steps. 

"""
PIECEWISE_STEPS_PASS1_LARGE = [
 ("random", dict(
        random_iterations=6000, nreturn=5,
        max_translation=22.0, max_rotation=0.07,
        max_scale=1.008, max_shear=0.01,
        alignment_spacing=6.0,
        metric="MMI", metric_args={"numberOfHistogramBins": 16},
        sampling="RANDOM", sampling_percentage=0.4,
        shrink_factors=(2, 1), smooth_sigmas=(1.8, 0.0),
    )),
    ("affine", dict(
        metric="MMI", metric_args={"numberOfHistogramBins": 16},
        optimizer="RSGD",
        optimizer_args=dict(learningRate=0.02, minStep=5e-7, numberOfIterations=200),
        sampling="RANDOM", sampling_percentage=0.25,
        shrink_factors=(4, 2, 1), smooth_sigmas=(2.0, 1.0, 0.0),
        initial_condition="IDENTITY",
    )),
]

BLOCKSIZE_PASS1 = (320, 320, 320)
OVERLAP_PASS1   = 0.35

"""
Outcome: current best, edgesare not perfectly snapping but some single cells matches are visible. 

Exp1: Testing a few Pass1 improvements: 
1. Increasing the max_scale and max_shear before moving to the next steps.
Res: This didn't yield any major improvements. 
2. Reducing the alignment_spacing to 3.0 um (from 6.0 um).
Res: This didn't yield any major improvements.
3. Splitting the stack along z to increase flexibility.
BLOCKSIZE_PASS1 = (20, 320, 320)
OVERLAP_PASS1   = 0.35
Res: Generated unatural bends in the tail. 
4. Romving the sampling: sampling="NONE",
Res: This didn't yield any major improvements.

--
Exp2: Adding a global deform step (runs on the cluster)

"""
DEFORM_A = ("deform", dict(
    control_point_spacing=300.0,
    control_point_levels=[1],
    metric="ANC", metric_args={"radius": 20},
    optimizer="RSGD", optimizer_args=dict(learningRate=0.006, minStep=8e-6, numberOfIterations=50),
    sampling="RANDOM", sampling_percentage=0.04,
    shrink_factors=(6, 3),
    smooth_sigmas=(8.0, 4.0),
))

DEFORM_B = ("deform", dict(
    control_point_spacing=250.0,
    control_point_levels=[1],
    metric="ANC", metric_args={"radius": 20},
    optimizer="RSGD", optimizer_args=dict(learningRate=0.008, minStep=5e-6, numberOfIterations=60),
    sampling="RANDOM", sampling_percentage=0.05,
    shrink_factors=(4, 2),
    smooth_sigmas=(6.0, 3.0),
))

DEFORM_C = ("deform", dict(
    control_point_spacing=200.0,
    control_point_levels=[1],
    metric="ANC", metric_args={"radius": 20},
    optimizer="RSGD", optimizer_args=dict(learningRate=0.010, minStep=4e-6, numberOfIterations=70),
    sampling="RANDOM", sampling_percentage=0.06,
    shrink_factors=(4, 2),
    smooth_sigmas=(5.0, 2.5),
))

DEFORM_D_SMALLER_GRID = ("deform", dict(
    control_point_spacing=160.0, control_point_levels=[1],
    metric="ANC", metric_args={"radius": 20},
    optimizer="RSGD",
    optimizer_args=dict(learningRate=0.006, minStep=8e-6, numberOfIterations=50),
    sampling="RANDOM", sampling_percentage=0.03,
    shrink_factors=(6, 3), smooth_sigmas=(8.0, 4.0),
))

# High-frequency (former test, aggressive)
DEFORM_E_GLOBAL_FINE = ("deform", dict(
    control_point_spacing=50.0, control_point_levels=[1],
    metric="ANC", metric_args={"radius": 20},
    optimizer="RSGD",
    optimizer_args=dict(learningRate=2.5, minStep=0.0, numberOfIterations=25),
    alignment_spacing=2.0,
    sampling="RANDOM", sampling_percentage=0.10,
    shrink_factors=(1,), smooth_sigmas=(0.25,),
))

# Safer global (low-frequency, short run)
DEFORM_F_GLOBAL_SAFE = ("deform", dict(
    control_point_spacing=120.0, control_point_levels=[1],
    metric="ANC", metric_args={"radius": 20},
    optimizer="RSGD",
    optimizer_args=dict(learningRate=0.08, minStep=3e-6, numberOfIterations=25),
    alignment_spacing=3.0,
    sampling="RANDOM", sampling_percentage=0.05,
    shrink_factors=(3, 1), smooth_sigmas=(2.0,),
))

"""
Outcome: 

"""