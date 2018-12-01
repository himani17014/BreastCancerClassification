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
#from data_preprocess impor
import numpy as np
import sys
from matplotlib.mlab import frange
from sklearn.metrics import roc_auc_score
import time

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
   if (TN+FP)!=0:
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
def svm_best_paramter_search(train_x,train_y):
    c_value=[0.001, 0.01, 0.1, 1, 10]
    gamma_value=[0.001, 0.01, 0.1, 1]
    parameters=[{'kernel':['linear','rbf'],'gamma':gamma_value,'C':c_value}]
    clf=GridSearchCV(SVC(probability=True),parameters,cv=5)
     	
    clf.fit(train_x,train_y)
   # print(clf.best_params_)
    return clf
#..........................................................................
      



if __name__ == '__main__':
    print("Runs as: python program_name pmat.csv")
    if len(sys.argv) != 2:  
       print("Provide correct number of arguments!!")
       sys.exit()
    else:   
       # Pmat=np.genfromtxt(sys.argv[1],unpack = True,delimiter=';')[:,:-1]
        Pmat=np.loadtxt(sys.argv[1],delimiter=' ',dtype=float)
        np.random.shuffle(Pmat)
             
          
        X=Pmat[:,0:Pmat.shape[1]-1]
        Y=Pmat[:,Pmat.shape[1]-1]
        
        
        trainX,testX,trainY,testY = train_test_split(X,Y,test_size=0.20,random_state=42)
        #print(trainX,trainY)         
         #..............call for Best paramter search.........................
        clf=svm_best_paramter_search(trainX,trainY)
                  
        print('Best C:',clf.best_estimator_.C) 
        print('Best Kernel:',clf.best_estimator_.kernel)
        print('Best Gamma:',clf.best_estimator_.gamma)    
        #....................................................................
        
        #...............storing Best parameters................................
        best_c=clf.best_estimator_.C
        best_kernel=clf.best_estimator_.kernel
        best_gamma=clf.best_estimator_.gamma
        #.......................................................................
        
        start_time=time.time()
        
        #................Creating model on best parameters.........................
        clf1=SVC(kernel=best_kernel,gamma=best_gamma,C=best_c,probability=True)
        #.........Train the model..................................................
        clf1.fit(trainX,trainY)
        
        end_time=time.time()
        print("Time spent in Training=",end_time-start_time)
        
        
        scores = cross_val_score(clf1, trainX, trainY, cv=10)
        print(scores)
        print("Average Cross validation accuracy=",np.sum(scores)/10)
        print("Training done on best parameters!!")
        
        
        #...........predict the probabilities for each test data point...............
        y_predicted_proba=clf1.predict_proba(testX)   # Finding predicted class probabbilities...(y_prob is numpy array)
        
        y_predicted_proba_final=   y_predicted_proba[:,1]
        #print("y_predicted_proba_final=",y_predicted_proba_final.shape)
                   
        y_true_class=testY.tolist()     #...True class labels
        
        
        #........writing output in file("Final_results.txt")...........................
              
        with open('Final_Results_svm.csv','w') as fp:
            fp.write("SVM_RESULTS\nThreshold Sensitivity FPR Specificity Accuracy MCC ROC_AUC_Value")
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
                
        print("Testing done and Results written to file:Final_Results_svm.csv")           
        #...........................END  of program..................................
         
        
        
       
        
     
        
#        clf = SVC(kernel='rbf')
#        clf.fit(trainX,trainY)
#        predY = clf.predict(testX)
#        
#        print ("\nAccuracy=",accuracy_score(testY,predY))

