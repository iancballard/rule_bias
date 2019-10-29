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
from trial_functions import check_abort, init_stims, present_dots_record_keypress, get_basic_objects
      

def draw_stim(win, stim, nframes):    
    for frameN in range(int(nframes)):
        stim.draw()
        win.flip()
        check_abort(event.getKeys())
          
def experiment_module(p, win):

    ########################
    #### Instructions ####
    ########################
    if p.step_num == 0:
        for txt in p.instruct_text['intro']:
            message = visual.TextStim(win,
                text=dedent(txt))
            message.draw()
            win.flip()
            keys = event.waitKeys(keyList  = ['1'])
                   
    else:
        for txt in p.instruct_text['break_txt']:
            txt = txt.replace('COMPLETED',str(p.step_num))
            txt = txt.replace('TOTAL',str(p.num_blocks))
            
            message = visual.TextStim(win,
                text=dedent(txt))
            message.draw()
            win.flip()                
            
    

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
    #### Set up Trial Order ####
    ############################
    
    p.ntrials = p.ntrials_per_miniblock * len(p.miniblocks)
    
    #create random color, shape, motion patterns for all trials
    p.dimension_val = {}
    p.dimension_correct_resp = {}
    for dimension in ['color','motion','shape']:
        
        if dimension == 'color':
            direction = ['green','pink'] * int(p.ntrials/2)
        elif dimension == 'shape':
            direction = ['circle','cross'] * int(p.ntrials/2)
        elif dimension == 'motion':
            direction = ['up','down'] * int(p.ntrials/2)

        correct_resp = ['1','2'] * int(p.ntrials/2)
        
        #shuffle
        resp = list(zip(direction, correct_resp))
        np.random.shuffle(resp)
        
        p.dimension_val[dimension], p.dimension_correct_resp[dimension] = zip(*resp)
     
    #set up miniblocks
    np.random.shuffle(p.miniblocks)
    ########################
    #### TODO Pseudorandomize miniblock structure ####
    ########################
    
    #create vector of correct response that implictly define "correct" rule
    p.correct_resp = []
    p.active_rule = []
    p.miniblock = []
    p.coherences = {}
    for block_num, block in enumerate(p.miniblocks):
        
        #create list of 'active' rules according to each miniblock
        block_rules = block.split('_') #2 active rules in a block
        block_rules = block_rules * int(p.ntrials_per_miniblock/2)
        np.random.shuffle(block_rules)
        
        ########################
        #### TODO make coherences are balanced for 'active'rule ####
        ########################
        
        #create coherences within miniblocks
        for dimension in ['color','motion','shape']:
        
            coherence = np.linspace(p.coherence_floor[dimension],
                                    p.coherence_floor[dimension] + .1,
                                    num=int(p.ntrials_per_miniblock/2))
            coherence = list(coherence)*2 #2 repeats
            np.shuffle(coherence)
            p.coherences[dimension] = coherence
        
            
        #get correct responses
        for n,rule in enumerate(block_rules):
            trial_idx = block_num*p.ntrials_per_miniblock + n
            resp = p.dimension_correct_resp[rule][trial_idx]
            
            p.correct_resp.append(resp)
            p.active_rule.append(rule)
            p.miniblock.append(block)
                
    
    ########################
    #### Run Experiment ####
    ########################
    
    # notify participant
    if p.step_num < 2: #after 2nd intro, "space to continue" is in instructions
        message = visual.TextStim(win,
            text='Press space to begin')
        message.draw()
        win.flip()

    
    #wait for scan trigger
    keys = event.waitKeys(keyList  = ['space'])

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
    p.incorrect = [] #only for switch
    p.bank = 0
    win.recordFrameIntervals = True
    num_correct= 0
    num_errors = 0
    
    for n in range(p.ntrials):
        
        ############################
        ###dot stim/choice period###
        ############################

        #set up response key and rt recording
        rt_clock = clock.getTime()
        p.choice_times.append(rt_clock)
        correct = True
        
        #color, motion and shape for this trial
        color = p.dimension_val['color'][n]
        motion = p.dimension_val['motion'][n]
        shape = p.dimension_val['shape'][n]
        rule = p.active_rule[n]
        print(color, motion, shape, rule)

        keys = present_dots_record_keypress(p,
                                            win,
                                            dotstims,
                                            cue,
                                            clock,
                                            color, shape, motion,
                                            rule)
        
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
            draw_stim(win,
                feedback_text['missed'],
                nframes)
            
        elif str(resp) != str(p.correct_resp[n]):
            correct = False
            draw_stim(win,
                feedback_text['incorrect'],
                nframes)
        
        if not correct:
            num_errors +=1   

        ################
        ###iti period###
        ################
        
        draw_stim(win,
                    fixation,
                    p.iti * win.framerate)
    

    print('errors',num_errors, num_errors/p.ntrials)
    print('mean_rt',np.nanmean(p.rt))
    print('\nOverall, %i frames were dropped.\n' % win.nDroppedFrames)

    #save data
    out_f = op.join(p.outdir,p.sub + '_switch_' + str(p.step_num) + '.pkl')
    with open(out_f, 'wb') as output:
        pickle.dump(p, output, pickle.HIGHEST_PROTOCOL)
        
                              
def main(arglist):

    ##################################
    #### Parameter Initialization ####
    ##################################
    
    # Get the experiment parameters
    mode = arglist.pop(0)
    p = datastruct.Params(mode)
    p.set_by_cmdline(arglist)
    
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

    for n in range(p.num_blocks):
        p.step_num = n
        c = experiment_module(p, win)
        
    core.quit()
   
    
if __name__ == "__main__":
   main(sys.argv[1:])    
   

