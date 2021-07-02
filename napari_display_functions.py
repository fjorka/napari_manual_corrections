# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 13:27:49 2021

@author: Kasia Kedziora
"""

import napari
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure

def display_set(viewer,im_track,im_labels,im_signal_list,colors_list,names_list,label_contour = 0):
    
    '''
    Function to create or update a viewer
        
    input: (= output of gallery_create_all)
        gallery_track
        gallery_labels
        gallery_signal_list - this is a list itself for all the signal images
    output:
        -
    '''
    try:
        viewer.layers['tracking channel'].data = im_track
    except KeyError:
        viewer.add_image(im_track,colormap='gray',contrast_limits=(0, 2000),opacity = 1,name = 'tracking channel')

    ################################
    for myIm_signal,myColor,myName in zip(im_signal_list,colors_list,names_list):
        
        try:
            # if the layer exists, update the data
            viewer.layers[myName].data = myIm_signal
        except KeyError:
            # otherwise add it to the viewer  
            viewer.add_image(myIm_signal,colormap=myColor,contrast_limits=(0, 4000),name = myName,opacity=0.5)
    
    ###############################
    try:
        viewer.layers['objects'].data = im_labels  
    except KeyError:
        viewer.add_labels(im_labels,name='objects',opacity = 0.5)
        viewer.layers['objects'].contour = label_contour
    
    return viewer


def create_graph_widget(track_intensity,signal_intensity_ch,colors_list,names_list,cellDataAll,myTrack):
    
    # select appropriate data
    myColumns=['t',track_intensity]
    myColumns.extend(signal_intensity_ch)
    myGraph_data = cellDataAll.loc[cellDataAll.track_id==myTrack,myColumns]
    myGraph_data = myGraph_data.sort_values('t')

    # create widget
    mpl_widget = FigureCanvas(Figure(tight_layout=True))

    ax_number = len(names_list)+1
    static_ax = mpl_widget.figure.subplots(ax_number,1)

    # set tracking channel
    static_ax[0].plot(myGraph_data.t,myGraph_data[track_intensity])
    static_ax[0].set_title('tracking channel',color='white')
    static_ax[0].tick_params(axis='x', colors='white')
    static_ax[0].tick_params(axis='y', colors='white')

    for i,col_name,myColor,myName in zip(range(len(names_list)),signal_intensity_ch,colors_list,names_list): 

        static_ax[i+1].plot(myGraph_data.t,myGraph_data[col_name],color=myColor)
        static_ax[i+1].set_title(myName,color='white')
        static_ax[i+1].tick_params(axis='x', colors='white')
        static_ax[i+1].tick_params(axis='y', colors='white')
        
        if col_name=='cyc_D_over_p21':
            static_ax[i+1].set_ylim(-5,5)
        
    return mpl_widget

