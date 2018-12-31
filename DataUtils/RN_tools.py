# -*- coding: utf-8 -*-

def find_chunks(nums, destroy_thr=.2, sec_thr=.5, sec_num=3):
    ''' 
       nums - A list of numbers corresponding to slices with predictions for a
              given scan.
       destroy_thr - A threshold (in decimal e.g. .2), that a chunk of sequential
              numbers should be removed if under.
       sec_thr - A secondary threshold, that a chunk of sequential numbers, if 
                 over the first destory_thr, but under sec_thr, should be checked
                 to see if the chunk is nearby another chunk, specifically within
                 sec_num slices away.
       sec_num - Param used with sec_thr, see above. 
    
    
    Create and return two lists, a list of slc nums to remove and to add.'''
    
    to_destroy = []
    to_add = set()
    
    nums.sort()
    chunks = []
    
    #Init first chunk
    chunk = [nums[0]]
    
    #Create chunks of sequential numbers
    for num in nums[1:]:
        if num == chunk[-1] + 1:
            chunk.append(num)
        else:
            chunks.append(chunk)
            chunk = [num]
    
    #Append the final chunk        
    chunks.append(chunk)
    
    #Calculate by percent the size of each chunk
    chunk_per = [len(chunks[i]) / len(nums) for i in range(len(chunks))]
    
    for c in range(len(chunks)):
        
        #If the percent size of the chunk is less then destroy_threshold, add to_destory
        if chunk_per[c] < destroy_thr:
            to_destroy += chunks[c]
        
        #If less then the secondary threshold, check chunks to either side
        elif chunk_per[c] < sec_thr:
            
            keep = False
            
            try:
                if chunk_per[c-1] > destroy_thr:
                    c1 = chunks[c-1][-1]
                    c2 = chunks[c][0]
                    
                    #If within sec_num away, keep everything between
                    if c2 - c1 < sec_num:
                        keep = True
                        
                        for x in range(c1+1,c2):
                            to_add.add(x)
            
            #In case this is the first chunk, pass as chunk_per[c-1] will throw error
            except:
                pass
            
            #Check other direction
            try:
                if chunk_per[c+1] > destroy_thr:
                    c1 = chunks[c][-1]
                    c2 = chunks[c+1][0]
                    
                    if c2 - c1 < sec_num:
                        keep = True
                        
                        for x in range(c1+1,c2):
                            to_add.add(x)
            
            #In case this is the last chunk, skip~
            except:
                pass
            
            #If the keep flag was not thrown, then remove this chunk
            if not keep:
                to_destroy += chunks[c]

    #Convert to_add to list, and ensure not adding any repeats from nums + sort
    to_add = [n for n in to_add if n not in nums]
    to_add.sort()
                
    return to_destroy, to_add

def post_process_boxes(data_points):
    ''' 
    data_points - A list containing the reference datapoints, assuming that
                  for each datapoint a predicted label of the form:
                  [x0, y0, x1, y1].
                  was assigned by a RN style network.
              
    Returns a modified version of the data_points, with post-proc. changes
    made to the pred_labels, specifically removing some preds and adding others.
    '''
    
    
    file_dict = {}
    indx_ref_list = [dp.get_ref() for dp in data_points]
    
    #For each datapoint, create a dict entry with all slices containing pred_label
    for dp in data_points:
        
        if len(dp.get_pred_label()) > 0:
        
            nm = dp.get_name()
            slc = dp.get_slc()
            
            try:
                file_dict[nm].append(slc)
            
            except KeyError:
                file_dict[nm] = [slc]
    
    #For each unique file, use find_chunks to find a list of slices to destory
    #and to add.
    for name in file_dict:
        to_destroy, to_add = find_chunks(file_dict[name])
        
        #For all labels within to_destory, set pred_label to []
        for d in to_destroy:
            ind = indx_ref_list.index(name+str(d))
            data_points[ind].set_pred_label([])
        
        #Nums to add assumes that we are adding/filling up to sec_num 
        #slices between two chunks of slices, can therefore assume that
        #a-1, represents a slice with a prediction. 
        for a in to_add:
            
            c_ind = indx_ref_list.index(name+str(a-1))
            ind = indx_ref_list.index(name+str(a))
            
            data_points[ind].set_pred_label(
                    data_points[c_ind].get_pred_label(copy=True)
                    )
            
    return data_points
            
            
        
    
        
    
    
    
    
