import psychopy
from psychopy import core, visual, event, logging
from psychopy.visual import ShapeStim, Polygon
from psychopy.monitors import Monitor
import socket
import json
import glob
import numpy as np
import os.path as op
import sys, getopt
import time
import pickle
import datastruct 
import pandas as pd
import seaborn as sns
from textwrap import dedent
from trial_functions import check_abort, init_stims, present_dots_record_keypress, get_basic_objects, update_rule_names
from psychopy import data
      

def draw_stim(win, stim, nframes):    
    for frameN in range(int(nframes)):
        stim.draw()
        win.flip()
        check_abort(event.getKeys())
        
#annoying function because setting opacity for textstim doesn't work    
def draw_error(win, nframes, fixation_color):
    for frameN in range(int(nframes)):    
        error = visual.TextStim(win,
                color = fixation_color,
                text='+',
                opacity = float((frameN % 8) >= 4))
        error.draw()
        win.flip()
        check_abort(event.getKeys())
                  
def experiment_module(p, win):

    ########################
    #### Instructions ####
    ########################
    update_rule_names(p)
    if p.step_num == 0:
        for txt in p.instruct_text['intro']:
            message = visual.TextStim(win,
                height = p.text_height,
                text=dedent(txt))
            message.draw()
            win.flip()
            keys = event.waitKeys(keyList  = ['space'])
                   
    else:
        for txt in p.instruct_text['break_txt']:
            txt = txt.replace('COMPLETED',str(p.step_num))
            txt = txt.replace('TOTAL',str(p.num_blocks))
            
            message = visual.TextStim(win,
                text=dedent(txt))
            message.draw()
            win.flip()                
            keys = event.waitKeys(keyList  = ['space'])
            
    

    ########################
    #### Visual Objects ####
    ########################
    
    #colors
    p.dot_colors = p.lch_to_rgb(p)
    
    #dotstims init
    dotstims, cue = init_stims(p, win)

    #get fixation cross and feedback info
    fixation, feedback_text = get_basic_objects(win, p)
       
    ############################
    #### Set up Staircasing ####
    ############################
    
    conditions=[
        # {'label':'easy', 'startVal': 0.6, 'stepSizes' : .01, 'stepType': 'lin', 'minVal': .51, 'maxVal': 1, 'nUp': 1, 'nDown': 5, 'nReversals': 15},
        {'label':'hard',
        'startVal': p.coherence[p.rule],
        'stepSizes' : .01,
        'stepType': 'lin',
        'minVal': p.min_value[p.rule],
        'maxVal': 1,
        'nUp': 1, 'nDown': 1,
        'nReversals': p.n_reversals},
        ]
    stairs = data.MultiStairHandler(conditions=conditions, nTrials=10)
    
    ########################
    #### Run Experiment ####
    ########################

    #start timer
    clock = core.Clock()   
    
    #draw fixation
    draw_stim(win,
                fixation,
                1 * win.framerate)

    p.resp = []
    p.rt = []
    p.choice_times = []
    p.feedback_times = []
    p.correct = []
    p.incorrect = [] #only for switch
    p.bank = 0
    win.recordFrameIntervals = True
    num_correct= 0
    num_errors = 0
    
    other_dimensions = [x for x in ['color','shape','motion'] if x != p.rule]
    
    for intensity, staircase in stairs:
        
        ############################
        ###dot stim/choice period###
        ############################
        
        p.coherence[p.rule] = intensity
        
        #initialize dots    
        dotstims, cue = init_stims(p, win)
        
        #set up response key and rt recording
        rt_clock = clock.getTime()
        p.choice_times.append(rt_clock)
        correct = True
        
        #color, motion and shape for this trial
        trial_features = {}
        for other_dim in other_dimensions:
            trial_features[other_dim] = np.random.choice(p.rule_features[other_dim])

        #get motion rule and correct response
        act_resp = list(zip(p.rule_features[p.rule],
                         ['1','2']))
        np.random.shuffle(act_resp)
        trial_features[p.rule], correct_resp = act_resp[0]
        
        #present trial
        keys = present_dots_record_keypress(p,
                                            win,
                                            dotstims,
                                            cue,
                                            clock,
                                            trial_features['color'], trial_features['shape'], trial_features['motion'],
                                            p.rule)
        
        #record keypress
        if not keys:
            resp = np.NaN
            p.resp.append(resp)
            p.rt.append(np.NaN)
        else:
            resp = keys[0][0]
            p.resp.append(resp)
            p.rt.append(keys[0][1] - rt_clock)  
        
        #####################
        ###feedback period###
        #####################
        
        #show feedback    
        nframes = p.feedback_dur * win.framerate
            
        if np.isnan(p.rt[-1]): 
            correct = False
            draw_error(win, nframes, p.too_slow_color)
            
        elif str(resp) != str(correct_resp):
            correct = False            
            draw_error(win, nframes, p.fixation_color)
        
        if not correct:
            num_errors +=1   
        
        p.correct.append(correct)
        
        ##update staircase handler
        stairs.addResponse(int(correct))
        stairs.addOtherData('rt', p.rt[-1])
        stairs.addOtherData('color', trial_features['color'])
        stairs.addOtherData('shape', trial_features['shape'])
        stairs.addOtherData('motion', trial_features['motion'])
                
        ################
        ###iti period###
        ################
        
        draw_stim(win,
                    fixation,
                    p.iti * win.framerate)
    
    
    print_reversal = min(p.n_reversals,6)
    
    print('mean reversal',p.rule, np.mean(stairs.staircases[0].reversalIntensities[-print_reversal:]))
    print('std reversal',p.rule, np.std(stairs.staircases[0].reversalIntensities[-print_reversal:]))
    print('mean_rt',np.nanmean(p.rt))
    print('\nOverall, %i frames were dropped.\n' % win.nDroppedFrames)

    #save data
    out_f = op.join(p.outdir,p.sub + '_psychophys_run' + str(p.run) + '_' + p.rule + '.pkl')
    while op.exists(out_f):
        out_f = out_f[:-4] + '+' + '.pkl'

    with open(out_f, 'wb') as output:
        pickle.dump(p, output, pickle.HIGHEST_PROTOCOL)
    
    # save data as multiple formats
    filename = op.join(p.outdir,p.sub + '_run' + str(p.run) + '_staircase_' + p.rule)
    stairs.saveAsExcel(filename)  # easy to browse
    stairs.saveAsPickle(filename)
        
                              
def main(arglist):

    ##################################
    #### Parameter Initialization ####
    ##################################
    
    # Get the experiment parameters
    mode = arglist.pop(0)
    p = datastruct.Params(mode)
    p.set_by_cmdline(arglist)
    p.randomize_shape_assignments()
    
    ##################################
    #### Window Initialization ####
    ##################################
    
    # Create a window
    win = p.launch_window(p)        
    logging.console.setLevel(logging.WARNING)
    
    #hide mouse
    event.Mouse(visible = False)
                  
    ########################
    #### Task Blocks ####
    ########################

    p.num_blocks = len(p.psychophys_blocks)
    for n, rule in enumerate(p.psychophys_blocks):
        p.step_num = n
        p.rule = rule
        c = experiment_module(p, win)
        
    core.quit()
   
    
if __name__ == "__main__":
   main(sys.argv[1:])    
   

