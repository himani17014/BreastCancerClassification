
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 15:57:25 2018

@author: dilip
"""


from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
import time
#from data_preprocess impor
import numpy as np
import sys
from matplotlib.mlab import frange
from sklearn.metrics import roc_auc_score

#..................Function to compute metric values for each threshold.........                
def find_performance_metrics(true_class,y_predicted_proba_final,thres):
   TP=0
   FP=0
   FN=0
   TN=0
      
   for i in range(len(true_class)):
        
        if true_class[i] == 1 and y_predicted_proba_final[i]>= thres:
            TP +=1
   for i in range(len(true_class)):

        if true_class[i] == 0 and y_predicted_proba_final[i] <thres:
            TN +=1  
      
   for i in range(len(true_class)):
   
        if true_class[i] == 1 and y_predicted_proba_final[i]< thres:
            FN +=1
   for i in range(len(true_class)):
        
        if true_class[i] == 0 and y_predicted_proba_final[i]>= thres:
            FP +=1  
   
   #...................find metrics
   total=TP+FP+FN+TN
   #print('total',total)

   if (TN+FN)!=0:
       FPR=FP/(FP+TN)
   else:
       FPR=0
   MCC_deno= (TN+FN)*(TP+FN)*(TN+FP)*(TP+FP)
   if (TP+FN)!=0:
       Sensitivity=(TP*100)/float(TP+FN)
   else:
       Sensitivity=0
   if (TN+FP)!=0:    
       Specificity=(TN*100)/float(TN+FP)
   else:
       Specificity=0
   Accuracy=((TP+TN)*100)/float(total)
   if MCC_deno!=0:
       MCC=(((TP*TN)-(FP*FN))/float(MCC_deno))**0.5
   else:
       MCC=0    
   
  
   return TP, FP, TN, FN,Sensitivity,FPR,Specificity,Accuracy , MCC
           
             
#.............................................................................            
    


#...........Fuction to compute best parameters...............................
def decision_best_paramter_search(train_x,train_y):
    param_grid = {"criterion": ["gini", "entropy"],
              "min_samples_split": [2, 10, 20],
              "max_depth": [2, 4, 6],
              "min_samples_leaf": [1, 5, 10],
              "max_leaf_nodes": [None, 5, 10, 20]
              }

    classifier = GridSearchCV(DecisionTreeClassifier(), param_grid)
    classifier.fit(train_x, train_y)
    #print(classifier.best_params_)
    return classifier
#..........................................................................
      



if __name__ == '__main__':
    
    print("Runs as: python program_name pmat.csv")
    if len(sys.argv) != 2:  
       print("Provide correct number of arguments!!")
       sys.exit()
    else:   
       # Pmat=np.genfromtxt(sys.argv[1],unpack = True,delimiter=';')[:,:-1]
         Pmat=np.loadtxt(sys.argv[1],delimiter=' ',dtype=float)
         # np.random.shuffle(Pmat)
             
          
         X=Pmat[:,0:Pmat.shape[1]-1]
         Y=Pmat[:,Pmat.shape[1]-1]
        
        
         trainX,testX,trainY,testY = train_test_split(X,Y,test_size=0.20,random_state=42)
        #print(trainX,trainY)         
         #..............call for Best paramter search.........................

         clf=decision_best_paramter_search(trainX,trainY)
                  
         print('Best criterion:',clf.best_estimator_.criterion) 
         print('Best depth:',clf.best_estimator_.max_depth)
         print('Best max_leaf_nodes:',clf.best_estimator_.max_leaf_nodes)  
         print('Best max_samples_leaf:',clf.best_estimator_.min_samples_leaf)
         print('Best max_samples_split:',clf.best_estimator_.min_samples_split)
         #....................................................................
        
         #...............storing Best parameters................................
         best_cr=clf.best_estimator_.criterion
         best_depth=clf.best_estimator_.max_depth
         best_max_leaf_nodes=clf.best_estimator_.max_leaf_nodes
         best_min_samples_leaf=clf.best_estimator_.min_samples_leaf
         best_min_samples_split=clf.best_estimator_.min_samples_split         
         
         #.......................................................................
        
         #................Creating model on best parameters.........................



         clf1=DecisionTreeClassifier(criterion=best_cr,max_depth=best_depth, 		
	     max_leaf_nodes=best_max_leaf_nodes,min_samples_leaf=best_min_samples_leaf,min_samples_split=best_min_samples_split)
         #.........Train the model..................................................
         start_time = time.time()
         clf1.fit(trainX,trainY)
        
         end_time=time.time()
         print("Time spent in Training=",end_time-start_time)
         scores = cross_val_score(clf1, trainX, trainY, cv=5)
         print(scores)
         print("Average Cross validation accuracy",np.sum(scores)/5)
         print("Training done on best parameters!!")
        
        
         #...........predict the probabilities for each test data point...............
         y_predicted_proba=clf1.predict_proba(testX)   # Finding predicted class probabbilities...(y_prob is numpy array)
        
         y_predicted_proba_final= y_predicted_proba[:,1]
         #print("y_predicted_proba_final=",y_predicted_proba_final.shape)
                   
         y_true_class=testY.tolist()     #...True class labels
        
         
         #........writing output in file("Final_results.txt")...........................
              
         with open('outputs/Final_results_decision.csv','w') as fp:
             fp.write("DECISION_RESULTS\nThreshold Sensitivity FPR Specificity Accuracy MCC ROC_AUC_Value")
             fp.write('\n') 
          
             for thres in frange(0.1,1.0,0.1):
                 TP,TN,FP,FN,Sensitivity,FPR,Specificity,Accuracy,MCC=find_performance_metrics(y_true_class,y_predicted_proba_final,thres)
                 fp.write('%.2f;' %thres)
                 fp.write('%.2f;'  %Sensitivity)
                 fp.write('%.2f;'  %FPR)
                 fp.write('%.2f;'  %Specificity)
                 fp.write('%.2f;'  %Accuracy )
                 fp.write('%.2f;'   %MCC)
                 fp.write('%.2f;'   %roc_auc_score(testY,y_predicted_proba_final))
                 fp.write('\n')
                
         print("Testing done and Results written to file:Final_results_decision.csv")
         #...........................END  of program..................................
         
        
        
       
        
     
        
#        clf = SVC(kernel='rbf')
#        clf.fit(trainX,trainY)
#        predY = clf.predict(testX)
#        
#        print ("\nAccuracy=",accuracy_score(testY,predY))


