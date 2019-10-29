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
                
    if p.step_num < 3: #run through instructions for each feature the first time
        for txt in p.instruct_text[p.training_step]:
            message = visual.TextStim(win,
                text=dedent(txt))
            message.draw()
            win.flip()
            keys = event.waitKeys(keyList  = ['1'])     
    else:
        for txt in p.instruct_text['break_txt']:
            txt = txt.replace('COMPLETED',str(p.step_num))
            txt = txt.replace('TOTAL',str(p.total_trials))
            txt = txt.replace('FEATURE',p.training_step)
            
            message = visual.TextStim(win,
                text=dedent(txt))
            message.draw()
            win.flip()                
            
    ########################
    #### Coherence #########
    ########################
    p.coherence_record[p.training_step] = []
    
    #slow down updates if we get close to the end of the range
    if p.coherence['motion'] < .15:
        p.coherence_update['motion'] = .01
        
    
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
    p.dimension_val = {}
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
        
        p.dimension_val[dimension], correct_resp = zip(*resp)
        
        if p.training_step == dimension:
            p.correct_resp = correct_resp
     
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
        color_idx = p.dimension_val['color'][n]
        motion_idx = p.dimension_val['motion'][n]
        shape_idx= p.dimension_val['shape'][n]
        print(color_idx, motion_idx, shape_idx)

        keys = present_dots_record_keypress(p,
                                            win,
                                            dotstims,
                                            cue,
                                            clock,
                                            color_idx, shape_idx, motion_idx,
                                            p.training_step)
        
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
 
        ######################
        ###update coherence###
        ######################   
        
        if correct:
            num_correct += 1
            if num_correct == 1:
                p.coherence[p.training_step] = p.coherence[p.training_step] - p.coherence_update[p.training_step]
                
                #don't let it get too low
                if p.training_step in ['color','shape']:
                    p.coherence[p.training_step] = max(p.coherence[p.training_step], .51) 
                num_correct = 0
                    
        else: #wrong
            p.coherence[p.training_step] = p.coherence[p.training_step] + p.coherence_update[p.training_step]            
            num_correct = 0
                
        p.coherence_record[p.training_step].append(p.coherence[p.training_step])
        print(p.coherence[p.training_step])
        ################
        ###iti period###
        ################
        
        draw_stim(win,
                    fixation,
                    p.iti * win.framerate)
    

    print('errors',num_errors, num_errors/p.ntrials)
    print('mean_rt',np.nanmean(p.rt))
    print('coherence', p.training_step, np.mean(p.coherence_record[p.training_step]))
    print('\nOverall, %i frames were dropped.\n' % win.nDroppedFrames)

    #save data
    out_f = op.join(p.outdir,p.sub + '_training_' + p.training_step + str(p.step_num) + '.pkl')
    with open(out_f, 'wb') as output:
        pickle.dump(p, output, pickle.HIGHEST_PROTOCOL)
        
    return np.mean(p.coherence_record[p.training_step])
                              
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
    # Open up the stimulus window
    win = p.launch_window(p)        
    logging.console.setLevel(logging.WARNING)
    
    #hide mouse
    event.Mouse(visible = False)
                  
    
    ########################
    #### Task Blocks ####
    ########################
    p.coherence_record = {}
    
    #loop through training steps
    color = []
    motion = []
    p.total_trials = len(p.training_blocks)
    for n,training_step in enumerate(p.training_blocks):
        p.training_step = training_step
        p.step_num = n
        c = experiment_module(p, win)
        # if training_step == 'color':
        #     color.append(c)
        # else:
        #     motion.append(m)
    
    # #output summary of performance
    # print('color_summary',color, 'motion_summary', motion)
    # for n in range(1,int(len(p.training_blocks)/2 + 1)):
    #     print('color mean last ' + str(n) + ' trials', np.mean(color[-n:]))
    #     print('motion mean last ' + str(n) + ' trials', np.mean(motion[-n:]))
        
    core.quit()
   
    
if __name__ == "__main__":
   main(sys.argv[1:])    
   

