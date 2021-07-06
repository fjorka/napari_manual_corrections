# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 11:34:01 2021

@author: Kasia Kedziora
"""

import importlib
import sys

import pandas as pd
import numpy as np
from skimage import measure

sys.path.append(r'D:\BARC\napari_tracking_manual')
fov_f = importlib.import_module('fovRingsLibrary')

def update_dataFrame(myIm,myIm_signal_list,myLabels,cellDataAll,myT,myLabel):
    
    '''
    Function to use viewer data to modify data frame with all data (for a specific object in a specific frame)
    
    input:
        myIm
        myIm_signal_list
        myLabels
        cellDataAll
        myT
        myLabel
    
    output:
       cellDataAll 
    '''
    
    # create intensity image
    intIm = create_intensityImage(myIm,myIm_signal_list,myT)

    # create mask with only a selected object
    single_label_im = create_singleLabel(myLabels,myT,myLabel)
    
    # characterize new nucleus
    cellData = characterize_newNucleus(single_label_im,intIm)
    
    # create ring image
    x = int(cellData['centroid-0'])
    y = int(cellData['centroid-1'])
    single_label_ring = make_ringImage(single_label_im,x,y,imSize=200)
    
    # measure properties of the ring
    ringData = characterize_newRing(single_label_ring,intIm)
    
    # put data frames together
    cellDataAll = mod_dataFrame(cellDataAll,cellData,ringData,myT)
    
    return cellDataAll


def create_singleLabel(myLabels,myT,myLabel):
    
    '''
    Function to create a label image containing only a single cell
    
    input:
        myLabels
        myT
        myLabel
    
    output:
       single_label_im 
    '''
    
    # create mask with only a selected object
    single_label_im = myLabels[myT,:,:].copy()
    single_label_im[single_label_im != myLabel]=0
    
    return single_label_im
    
def create_intensityImage(myIm,myIm_signal_list,myT):

    '''
    Function to create intensity image for calculation from a single object
    
    input:
        myLabels
        myT
        myLabel
    
    output:
       intIm 
    '''
    
    intIm = [myIm[myT,:,:]]
    intIm.extend([x[myT,:,:] for x in myIm_signal_list])
    
    intIm = np.array(intIm)
    intIm = np.moveaxis(intIm,0,2)
    
    return intIm
    

def characterize_newNucleus(single_label_im,intIm):
    
    '''
    Function to get properties of a single cell
    
    input:
        single_label_im
        intIm
    
    output:
        cellData - data frame with regionprops of a single object    
    '''
    # define properties to calculate
    regProps = ['label', 'area','centroid','major_axis_length','minor_axis_length','orientation','bbox','image','mean_intensity']
    
    # find features of the new object
    cellData = measure.regionprops_table(single_label_im, properties=regProps,intensity_image=intIm)
    cellData = pd.DataFrame(cellData)
    
    return cellData

def make_ringImage(single_label_im,x,y,imSize=200):
    
    '''
    Function to get properties of a single cell
    
    input:
        single_label_im
    
    output:
        single_label_ring  
    '''
    
    myFrame = int(imSize/2)
    
    # cut small image
    small_im = single_label_im[x-myFrame:x+myFrame,y-myFrame:y+myFrame]
    
    # change small image into a ring
    rings = fov_f.make_rings(small_im,width=6,gap=1)
    
    # put small rings image back into the whole frame
    single_label_ring = single_label_im.copy()
    single_label_ring[x-myFrame:x+myFrame,y-myFrame:y+myFrame]=rings
    
    return single_label_ring
    

def characterize_newRing(single_label_ring,intIm):
    
    '''
    Function to get properties of a single cell
    
    input:
        single_label_im
        intIm
    
    output:
        cellData - data frame with regionprops of a single object    
    '''
    # define properties to calculate
    properties_ring = ['label','centroid','mean_intensity']
    
    # find features of the new object
    ringData = measure.regionprops_table(single_label_ring, properties=properties_ring,intensity_image=intIm)
    ringData = pd.DataFrame(ringData)
    
    return ringData

def mod_dataFrame(cellDataAll,cellData,ringData,myT):
    
    '''
    function to modify gneral data frame with updated modified single object data
    
    input:
        cellDataAll - original general data frame
        cellData
        ringData
        myT
        
    output:
        cellDataAll - modified general data frame
    '''
    
    # check which cell it is
    myLabel = list(cellData['label'])[0]

    # put nucleus and ring data together
    cellData = pd.merge(cellData,ringData,how='inner',on='label',suffixes=('_nuc', '_ring'))
    
    # modify names
    myNames = list(cellData.columns)
    myNames[2] = 'centroid-0'
    myNames[3] = 'centroid-1'
    cellData.columns = myNames
    
    # add aditional info
    cellData['t'] = myT
    cellData['track_id'] = myLabel
    
    # collect information about this label and this time point to calculate 
    info_track = cellDataAll.loc[:,['track_id','parent','root','generation','accepted']].drop_duplicates()
    info_frame = cellDataAll.loc[cellDataAll.t==myT,['t','background_01','background_02','background_03','background_04']].drop_duplicates()
    
    # merge it to the data of this frame
    cellData = cellData.merge(info_track,on='track_id')
    cellData = cellData.merge(info_frame,on='t')
    
    # calculate corrected signals
    for ch in np.arange(1,5):
    
        cellData[f'intensity_{str(ch).zfill(2)}_nuc_corr'] = cellData[f'mean_intensity-{ch-1}_nuc'] - cellData[f'background_{str(ch).zfill(2)}']
        cellData[f'intensity_{str(ch).zfill(2)}_ring_corr'] = cellData[f'mean_intensity-{ch-1}_ring'] - cellData[f'background_{str(ch).zfill(2)}']

    # swap in the general data frame
    cellDataAll.drop(cellDataAll[((cellDataAll.t==myT) & (cellDataAll.track_id==myLabel))].index,axis=0,inplace=True)
    cellDataAll = cellDataAll.append(cellData,ignore_index=True)
    
    return cellDataAll

def mod_trackLayer(data,properties,cellDataAll,myT,myLabel):
    
    '''
    function to modify tracking layer for the viewer
    
    input:
        data
        properties
        cellDataAll
        myT
        myLabel
        
    output:
        data
        properties
    '''
    # choose the data for the specific object
    selData = cellDataAll.loc[((cellDataAll.t == myT) & (cellDataAll.track_id == myLabel)),:]
    
    # prepare in the right format
    frameData = np.array(selData.loc[:,['label','t','centroid-0','centroid-1']])
    
    # find position of this cell in the tracking data structure
    changeIndex = ((data[:,1]==myT) & (data[:,0]==myLabel))
    
    # change data
    data = np.delete(data,changeIndex,axis=0)
    data = np.vstack([data, frameData])
    
    # modify properties of the track layer

    selData.loc[:,'state'] = 5
    
    for tProp in properties.keys():
    
        properties[tProp] = np.delete(properties[tProp],changeIndex)
        properties[tProp] = np.append(properties[tProp], selData[tProp])
    
    return data, properties

def newTrack_number(vector):
    
    '''
    Function to find the smallest unused number for a track that can be used
    
    input:
        
        vector - array like with numbers used for tracks
        
    output:
        
        newTrack - number to be used for a new track
    
    '''
    # find number of independent tracks
    tracksSetLength = len(set(vector))
    
    # find maximum track number
    trackMax = np.max(vector)
    
    # check if all are used
    if (trackMax >= (tracksSetLength+1)):
        
        unusedTracks = set(vector).symmetric_difference(np.arange(trackMax+1))
        unusedTracks = np.array(list(unusedTracks))
        unusedTracks = unusedTracks[unusedTracks>0][0]
        newTrack = np.nanmin(unusedTracks)
        
    else:
        newTrack = trackMax + 1 
    
    return newTrack

def trackData_from_df(cellDataAll):
    
    '''
    Function to extract tracking data from a data frame
    
    input:
        cellDataAll - sorted
    
    output:
        data
        properties

    '''

    #############################################
    # prepare data
    #############################################
    
    # avoid objects without tracking data
    selVector = (cellDataAll.track_id==cellDataAll.track_id)
    
    #gather data in a form of numpy array
    data = np.array(cellDataAll.loc[selVector,['track_id','t','centroid-0','centroid-1']])
    
    # change format of tracks id
    data[:,0]=data[:,0].astype(int)


    #############################################
    # prepare properties
    #############################################
    # specify columns to extract properties
    properties = {}
    prop_prop = ['t', 'generation', 'root', 'parent', 'minor_axis_length', 'major_axis_length', 'area']
    
    for tProp in prop_prop:
    
        properties[tProp] = cellDataAll.loc[selVector,tProp]
    
    properties['state'] = [5]*len(properties['t'])
    
    #############################################
    # prepare graph
    #############################################
    graph = cellDataAll.loc[(~(cellDataAll.track_id == cellDataAll.parent) & selVector),['track_id','parent']].drop_duplicates().to_numpy()
    
    graph=graph.astype(int)
    graph = dict(graph)
    
    return data,properties,graph
    
def find_all_paths(graph, node, path=[]):
    
    '''
    Function to find all the paths coming through a node in a graph 
    
    input:
        graph
        node
    output:
        list of paths
    '''
    
    path = path + [node]
    paths = [path]
    
    offspring_list = []
    for key, value in graph.items():   # iter on both keys and values
            if (value == [node]):
                offspring_list.append(key)
    
    for node in offspring_list:
        newpaths = find_all_paths(graph, node, path)
        for newpath in newpaths:
            paths.append(newpath)
            
    return paths

def forward_labels(myLabels,cellDataAll,myT,myLabel,newTrack):
    
    '''
    Function to modify labels layer.
    input:
        myLabels
        cellDataAll
        myT
        myLabel
        newTrack
    output:
        myLabels
    '''

    for myInd in cellDataAll.index[(cellDataAll.track_id==myLabel) & (cellDataAll.t>=myT)]:
        
        row_start = cellDataAll.loc[myInd,'bbox-0']
        row_stop = cellDataAll.loc[myInd,'bbox-2']
        column_start = cellDataAll.loc[myInd,'bbox-1']
        column_stop = cellDataAll.loc[myInd,'bbox-3']
        
        if np.isnan(row_start and row_stop and column_start and column_stop):
            
            pass
        
        else:
            
            myFrame = int(cellDataAll.loc[myInd,'t'])
    
            # cut and replace
            temp = myLabels[myFrame,int(row_start):int(row_stop),int(column_start):int(column_stop)]
            temp[temp == myLabel] = int(newTrack)
            myLabels[myFrame,int(row_start):int(row_stop),int(column_start):int(column_stop)] = temp
    
    return myLabels

def forward_df(cellDataAll,myT,myLabel,newTrack,connectTo=0):
    
    '''
    Function to modify forward data frame structure after linking changes
    input:
        cellDataAll
        myT
        myLabel
        newTrack
        graph
    output:
        cellDataAll
    '''
    
    # find info about the cut track
    myLabel_generation = list(cellDataAll.loc[cellDataAll.track_id==myLabel,'generation'].drop_duplicates())[0]
    
    # find info about the new label
    genList = list(cellDataAll.loc[cellDataAll.track_id==newTrack,'generation'].drop_duplicates())
    if len(genList)>0:
        new_generation = genList[0]
        new_root = list(cellDataAll.loc[cellDataAll.track_id==newTrack,'root'].drop_duplicates())[0]
        new_parent = list(cellDataAll.loc[cellDataAll.track_id==newTrack,'parent'].drop_duplicates())[0]
        
    else: # so this is a completely new number for a track
        
        if connectTo == 0: # and nothing to connect to
            
            new_generation = 0
            new_root = newTrack
            new_parent = newTrack
        
        else: # check data of a track we connect to
        
            new_generation = list(cellDataAll.loc[cellDataAll.track_id==connectTo,'generation'].drop_duplicates())[0] + 1
            new_root = list(cellDataAll.loc[cellDataAll.track_id==connectTo,'root'].drop_duplicates())[0]
            new_parent = connectTo
            
    
    # get a graph
    data,properties,graph = trackData_from_df(cellDataAll)
    
    # find kids
    kids_list = []
    for key, value in graph.items():   # iter on both keys and values
            if (value == [myLabel]):
                kids_list.append(key)
    
    # find all family members
    all_paths = find_all_paths(graph, myLabel)
    family_members = [item for sublist in all_paths for item in sublist]
    
    for myDescendant in family_members:
        
        # find which rows need to be changed
        changeIndex = (cellDataAll.t>=myT) & (cellDataAll.track_id==myDescendant)
        
        cellDataAll.loc[changeIndex,'root'] = new_root
        cellDataAll.loc[changeIndex,'generation'] = cellDataAll.loc[changeIndex,'generation'] - myLabel_generation + new_generation
        
        if(myDescendant == myLabel):
        
            cellDataAll.loc[changeIndex,'track_id'] = newTrack
            cellDataAll.loc[changeIndex,'parent'] = new_parent
              
        elif (myDescendant in kids_list): #2nd generation
            
            cellDataAll.loc[changeIndex,'parent'] = newTrack
            
    return cellDataAll
    