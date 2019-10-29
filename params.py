from copy import deepcopy
import os.path as op
from textwrap import dedent
import seaborn as sns
import numpy as np

#base parameters for all runs
base = dict(
    decision_dur = 2,
    feedback_dur = 1,
    outdir = op.abspath('./data/'),
    circle_radius = 100,
    coherence = {'motion': .9,
                'color':.9,
                'shape':.9},             
    window_color = '#0a0a0a',#'#545454',
    full_screen = True,
    test_refresh = True,
    update_coherence = False, #adjust coherences according to rts
    monitor_units = 'deg',
    fixation_color = -.2
)

#dot parameters
base.update(
    dot_size = .13, #degrees of visual angle
    dot_aperture = 5, #degrees of visual angle
    dot_density = 24,
    motion_direction_map = {'up':270, 'down':90},
    chroma = 50,
    lightnesses = [80, 80],
    hues = [160, 340],
)

#polygonal cue parameters
base.update(
    poly_radius=.3,
    poly_linewidth=1,
    poly_color=-.2,
    cues = dict(shape = 3,
                      color = 4,
                      motion = 5)
    )

#feedback parameters
base.update(
    fb_circle_radius = 1,
    fb_text_size = .75,
    fb_left_loc = (-2.5,0),
    fb_right_loc = (2.5,0),
    fb_pos_color = '#997020',#'#E2A429',
    fb_neg_color = '#992020',#'#AA0707',
    fb_text_color = '#999999'
    )

train = deepcopy(base)

train.update(

    run_type = 'train',
    monitor_name = 'mbpro',
    ntrials = 5,#20,
    iti = 1,
    training_blocks = ['shape','motion','color']*5,#,'motion','color']*20,
    coherence_update = {'motion': .02,
                'color':.01,
                'shape':.01},
    coherence_step = [0]*3,
    feedback_dur = .75,
    decision_dur = 2,
    instruct_text = {
    'intro':
                    ["""
                    Welcome to the experiment.
                    Today you will be making decisions based on
                    arrays of colored, moving dots.
                    You can advance through the instructions by pressing the '1' key.
                    """,
                    """
                    In this part of the experiment, you will practice responding
                    based on the color and the motion of the dots.
                    You have 2 seconds to respond on each trial.
                    Try to respond quickly, but without sacrificing accuracy. 
                    """],
    'break_txt':    ["""
                    You have completed COMPLETED out of TOTAL trials.
                    Next you will respond based on FEATURE.
                    Take a short break if you'd like.
                    Press space to continue.
                    """],
    'motion':       ["""
                    In the following trials,
                    indicate the direction that most 
                    of the dots are moving in. 
                    Press '1' for up and '2' for down. 
                    """],
    'shape':       ["""
                    In the following trials,
                    indicate the most common shape.
                    Press '1' for circle and '2' for crosses. 
                    """],
    'color':        ["""
                    In the following trials,
                    indicate the most common
                    color of the dots. 
                    Press '1' for green and '2' for pink. 
                    """]},
    instruct_color = '#7E2BE0'
)

switch = deepcopy(base)

switch.update(

    run_type = 'train',
    monitor_name = 'mbpro',
    ntrials = 20,
    iti = 1,
    num_blocks = 10,
    miniblocks = ['color_shape','shape_motion','motion_color']*2,
    ntrials_per_miniblock = 10, #=60 trials per block
    feedback_dur = .75,
    decision_dur = 2,
    instruct_text = {
    'intro':
                    [
                    """
                    In this part of the experiment, you will practice responding
                    based on the color and the motion of the dots.
                    You have 2 seconds to respond on each trial.
                    Try to respond quickly, but without sacrificing accuracy. 
                    """],
    'break_txt':    ["""
                    You have completed COMPLETED out of TOTAL blocks.
                    Take a short break if you'd like.
                    Press space to continue.
                    """]
                },
    instruct_color = '#7E2BE0'
)

