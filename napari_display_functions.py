# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 13:27:49 2021

@author: Kasia Kedziora
"""

import napari
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure

from scipy.spatial import distance_matrix
import numpy as np

import importlib
gen = importlib.import_module('general_functions')

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

def cut_track(viewer,cellDataAll):
    
    '''
    Function to cut a track at a given point.
    '''
    
    # get images of objects
    myLabels = viewer.layers['objects'].data

    # get the position in time
    myT = viewer.dims.current_step[0]

    # get my label
    myLabel = viewer.layers['objects'].selected_label

    # find new track number
    newTrack = gen.newTrack_number(cellDataAll.track_id)

    #####################################################################
    # change labels layer
    #####################################################################

    myLabels = gen.forward_labels(myLabels,cellDataAll,myT,myLabel,newTrack)    
    viewer.layers['objects'].data = myLabels

    #####################################################################
    # modify data frame
    #####################################################################
    cellDataAll = gen.forward_df(cellDataAll,myT,myLabel,newTrack)

    #####################################################################
    # change tracking layer
    #####################################################################

    # modify the data for the layer
    data,properties,graph = gen.trackData_from_df(cellDataAll)

    # change tracks layer
    viewer.layers['tracking'].data = data
    viewer.layers['tracking'].properties = properties
    viewer.layers['tracking'].graph = graph

    #####################################################################
    # change viewer status
    #####################################################################
    viewer.status = f'Track {myLabel} was cut at frame {myT}.' 
    
    return viewer,cellDataAll

def merge_track(viewer,cellDataAll):
    
    '''
    Function to merge a track with a chosen track or the closest track in the previous frame
    
    input:
        viewer
        cellDataAll
    output:
        
    '''
    # get images of objects
    myLabels = viewer.layers['objects'].data
    
    # get the position in time
    myT = viewer.dims.current_step[0]
    
    # get my label
    myLabel = viewer.layers['objects'].selected_label
    
    if myT>0:
        
        connTrack=0
        
        # check if there is a point to merge too
        merge_to = viewer.layers['modPoints'].data
        
        if len(merge_to)==1:
            
            merge_to = merge_to[0]
            
            if merge_to[0] == (myT-1):
                
                connTrack = myLabels[tuple(merge_to.astype(int))]
                
                viewer.layers['modPoints'].data = []
                
            else:
                viewer.status = 'Merging cell does not match'
        
        elif len(merge_to)==0:
    
            # find the closest object in the previous layer
            object_data = cellDataAll.loc[((cellDataAll.track_id == myLabel) & (cellDataAll.t == myT)),['centroid-0','centroid-1']].to_numpy()
    
            candidate_objects = cellDataAll.loc[(cellDataAll.t == (myT-1)),['track_id','centroid-0','centroid-1']]
            candidate_objects_array = candidate_objects.loc[:,['centroid-0','centroid-1']].to_numpy()
    
            dist_mat = distance_matrix(object_data,candidate_objects_array)
            iloc_min = np.nanargmin(dist_mat)
    
            connTrack = int(candidate_objects.iloc[iloc_min,:].track_id)
            
            
        else:
            viewer.status = 'Only one point is allowed for merging.'
            
        if connTrack > 0:
            
            # check if there is another branch that needs to be cleaned
            deadBranch = cellDataAll.loc[((cellDataAll.track_id==connTrack) & (cellDataAll.t>=myT)),:]
            
            if len(deadBranch) > 0:
                
                # find new track number
                newTrack = gen.newTrack_number(cellDataAll.track_id)
                
                # modify labels
                myLabels = gen.forward_labels(myLabels,cellDataAll,myT,connTrack,newTrack)    
                
                # modify data frame
                cellDataAll = gen.forward_df(cellDataAll,myT,connTrack,newTrack)
                
    
            #####################################################################
            # change labels layer
            #####################################################################
    
            myLabels = gen.forward_labels(myLabels,cellDataAll,myT,myLabel,connTrack)    
            viewer.layers['objects'].data = myLabels
    
            #####################################################################
            # modify data frame
            #####################################################################
            cellDataAll = gen.forward_df(cellDataAll,myT,myLabel,connTrack)
    
            #####################################################################
            # change tracking layer
            #####################################################################
    
            # modify the data for the layer
            data,properties,graph = gen.trackData_from_df(cellDataAll)
    
            # change tracks layer
            viewer.layers['tracking'].data = data
            viewer.layers['tracking'].properties = properties
            viewer.layers['tracking'].graph = graph
            
    
            viewer.status = f'Track {myLabel} was merged with {connTrack}.'
            
    else:
        viewer.status = 'It is not possible to merge objects from the first frame.'
    
        
    return viewer,cellDataAll

def connect_track(viewer,cellDataAll):
    
    # developing connecting function
    
    # get images of objects
    myLabels = viewer.layers['objects'].data
    
    # get the position in time
    myT = viewer.dims.current_step[0]
    
    # get my label
    myLabel = viewer.layers['objects'].selected_label
    
    if myT>0:
    
        connTrack=0
    
        # check if there is a point to merge too
        merge_to = viewer.layers['modPoints'].data
    
        if len(merge_to)==1:
    
            merge_to = merge_to[0]
    
            if merge_to[0] == (myT-1):
    
                connTrack = myLabels[tuple(merge_to.astype(int))]
    
                viewer.layers['modPoints'].data = []
    
            else:
                viewer.status = 'Connecting cell does not match'
    
        elif len(merge_to)==0:
    
            # find the closest object in the previous layer
            object_data = cellDataAll.loc[((cellDataAll.track_id == myLabel) & (cellDataAll.t == myT)),['centroid-0','centroid-1']].to_numpy()
    
            candidate_objects = cellDataAll.loc[(cellDataAll.t == (myT-1)),['track_id','centroid-0','centroid-1']]
            candidate_objects_array = candidate_objects.loc[:,['centroid-0','centroid-1']].to_numpy()
    
            dist_mat = distance_matrix(object_data,candidate_objects_array)
            iloc_min = np.nanargmin(dist_mat)
    
            connTrack = int(candidate_objects.iloc[iloc_min,:].track_id)
    
    
        else:
            viewer.status = 'Only one mother object is allowed to be connected.'
    
        if connTrack > 0:
    
            # check if there is another branch that needs to be cleaned
            sisterBranch = cellDataAll.loc[((cellDataAll.track_id==connTrack) & (cellDataAll.t>=myT)),:]
    
            if len(sisterBranch) > 0:
    
                # find new track number
                newTrack_sister = gen.newTrack_number(cellDataAll.track_id)
    
                # modify labels
                myLabels = gen.forward_labels(myLabels,cellDataAll,myT,connTrack,newTrack_sister)    
    
                # modify data frame
                cellDataAll = gen.forward_df(cellDataAll,myT,connTrack,newTrack_sister,connectTo=connTrack)
    
    
            #####################################################################
            # change labels layer
            #####################################################################
            
            # find new track number
            newTrack = gen.newTrack_number(cellDataAll.track_id)
    
            myLabels = gen.forward_labels(myLabels,cellDataAll,myT,myLabel,newTrack)    
            viewer.layers['objects'].data = myLabels
    
            #####################################################################
            # modify data frame
            #####################################################################
            cellDataAll = gen.forward_df(cellDataAll,myT,myLabel,newTrack,connectTo=connTrack)
    
            #####################################################################
            # change tracking layer
            #####################################################################
    
            # modify the data for the layer
            data,properties,graph = gen.trackData_from_df(cellDataAll)
    
            # change tracks layer
            viewer.layers['tracking'].data = data
            viewer.layers['tracking'].properties = properties
            viewer.layers['tracking'].graph = graph
    
    
            viewer.status = f'Track {myLabel} was merged with {connTrack}.'
    
    else:
        viewer.status = 'It is not possible to connect objects from the first frame.'
        
    return viewer,cellDataAll

def update_single_object(viewer,cellDataAll,myIm,myIm_signal_list):
    
    # get images of objects
    myLabels = viewer.layers['objects'].data
    
    ########################################################
    # modify data frame
    ########################################################

    # get the position in time
    myT = viewer.dims.current_step[0]

    # get my label
    myLabel = viewer.layers['objects'].selected_label

    # calculate features of a new cell and store in the general data frame
    cellDataAll = gen.update_dataFrame(myIm,myIm_signal_list,myLabels,cellDataAll,myT,myLabel)

    # add additional columns
    # this is highly specific for an experiment, make more visible
    cellDataAll['DHB_ratio'] = cellDataAll['intensity_02_ring_corr']/cellDataAll['intensity_02_nuc_corr']
    cellDataAll['cyc_D_over_p21'] = cellDataAll['intensity_03_nuc_corr']/cellDataAll['intensity_04_nuc_corr']

    ########################################################
    # modify tracking layer
    ########################################################

    # modify the data for the layer
    data,properties,graph = gen.trackData_from_df(cellDataAll)

    # change tracks layer
    viewer.layers['tracking'].data = data
    viewer.layers['tracking'].properties = properties
    viewer.layers['tracking'].graph = graph

    ########################################################
    # change viewer status
    ########################################################
    viewer.status = f'Frame {myT} was modified.' 
    
    return viewer,cellDataAll