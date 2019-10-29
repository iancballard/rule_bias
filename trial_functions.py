from psychopy.visual import ShapeStim, Polygon
from psychopy import core, visual, event, logging
import dots

def check_abort(keys):
    if 'escape' in keys:
        core.quit()
        
def init_stims(p, win):
    #dots
    shape_list = ['cross','circle']
    color_list = ['green','pink']
    dotstims = {}
    for color_idx, color in enumerate(color_list):
        for shape in shape_list:
            dotstims[color + '_' + shape] = [
            dots.RandomDotMotion(win, ##high color coherence high shape coherence 
                                    color = p.dot_colors[color_idx],
                                    size = p.dot_size,
                                    shape = shape,
                                    density = p.dot_density * p.coherence['color'] * p.coherence['shape'] ,
                                    aperture = p.dot_aperture),
            dots.RandomDotMotion(win, ##low color coherence high shape coherence 
                                    color = p.dot_colors[1 - color_idx],
                                    size = p.dot_size,
                                    shape = shape,
                                    density = p.dot_density * (1 - p.coherence['color']) * p.coherence['shape'] ,
                                    aperture = p.dot_aperture),             
            dots.RandomDotMotion(win, ##high color coherence low shape coherence 
                                    color = p.dot_colors[color_idx],
                                    size = p.dot_size,
                                    shape = [x for x in shape_list if x != shape][0], #other shape
                                    density = p.dot_density * p.coherence['color'] * (1 - p.coherence['shape']),
                                    aperture = p.dot_aperture),
            dots.RandomDotMotion(win, ##low color coherence low shape coherence 
                                    color = p.dot_colors[1 - color_idx],
                                    size = p.dot_size,
                                    shape = [x for x in shape_list if x != shape][0], #other shape
                                    density = p.dot_density * (1 - p.coherence['color']) * (1 - p.coherence['shape']),
                                    aperture = p.dot_aperture)]
    
    #polygonal cue
    cue = Polygon(win,
                    radius=p.poly_radius,
                    lineColor=p.poly_color,
                    fillColor=p.poly_color,
                    lineWidth=p.poly_linewidth)
    
    
    return dotstims, cue
    

def present_dots_record_keypress(p, win, dotstims, cue, clock, color, shape, motion, rule):
    keys = False
    
    #polygonal cue
    cue.setEdges(p.cues[rule])
    cue.draw()

    #randomly initialize dot locations
    for ds in dotstims[color + '_' + shape]:
        ds.reset()
    win.flip()

    #loop through frames
    nframes = p.decision_dur * win.framerate
    for frameN in range(nframes): #update dot position
        #loop through 4 component dotstims
        for ds in dotstims[color + '_' + shape]:
            ds.update(p.motion_direction_map[motion],
                                        p.coherence['motion'])
            ds.draw()

        #draw cue
        cue.setEdges(p.cues[rule])
        cue.draw()

        win.flip()

        #detect keypresses
        if not keys: #only record first
            keys = event.getKeys(keyList = ['1','2'],
                                timeStamped = clock)  # get keys from event buffer
        
    return keys
     
def get_basic_objects(win, p):
    
    #fixation cross
    fixation = visual.TextStim(win,
        color = p.fixation_color,
        text='+')

    #feedback text 
    feedback_text = dict()
                        
    feedback_text['missed'] = visual.TextStim(win,
        color = '#af392a',
        text='Too slow')
        
    feedback_text['incorrect'] = visual.TextStim(win,
        color = '#af392a',
        text='Incorrect')
    
    return fixation, feedback_text