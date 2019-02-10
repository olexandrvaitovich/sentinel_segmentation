def prepare_data(x_folder, y_folder):
    
    masks = glob.glob("{}/*.tif".format(y_folder))
    images = glob.glob("{}/*.tif".format(x_folder))

    
    listx = []
    listy = []
    
    img_num = len(images)
    
    for i in range(img_num):
        
        mask = cv2.imread(masks[i])
        
        not_empty = np.count_nonzero(mask)
        
        if not_empty>2125:
            
            listy+=[mask]
            listx+=[cv2.imread(images[i])]
            
            not_empty = 0
    return np.array(listx)/255, np.delete(np.array(listy), [1, 2], 3)>240

def final_data():
    x, y = prepare_data("xtrain", "ytrain")
    x2, y = prepare_data("x2train", "ytrain")
    x3, y = prepare_data("x3train", "ytrain")
    x4, y = prepare_data("x4train", "ytrain")
    xx = np.concatenate((x, np.delete(x2, [1,2], 3)), axis=3)
    xx2 = np.concatenate((x3, np.delete(x4, [1,2], 3)), axis=3)
    xxx = np.concatenate((xx, xx2), axis=3)
    xtest = xxx[int(xxx.shape[0]*0.9)+1:]
    xtrain = xxx[:int(xxx.shape[0]*0.9)]
    ytest = y[int(y.shape[0]*0.9)+1:]
    ytrain = y[:int(y.shape[0]*0.9)]
    return (xtrain, ytrain, xtest, ytest)