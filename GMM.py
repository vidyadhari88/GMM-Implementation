
# coding: utf-8

# In[110]:


import cv2
import numpy as np
from matplotlib import pyplot as plt
import random
from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse
UBIT = "50290085"
np.random.seed(sum([ord(c) for c in UBIT]))


# In[111]:



# In[112]:


def computeMean(XcoOrdinate,YcoOrdinate):
    xCentro = 0
    yCentro = 0
    for i in range(0,len(XcoOrdinate)):
        xCentro = xCentro+XcoOrdinate[i]
        yCentro = yCentro+YcoOrdinate[i]
    xCentro = xCentro / len(XcoOrdinate)
    yCentro = yCentro / len(YcoOrdinate)
    
    return np.around(xCentro,5),np.around(yCentro,5)
    
    
def plottingGraph(list1X,list1Y,list2X,list2Y,list3X,list3Y,cent1X,cent1Y,cent2X,cent2Y,cent3X,cent3Y):
    
    
    plt.scatter(list1X,list1Y,marker='^',c='red', s=50)
    plt.scatter(list2X,list2Y,marker='^',c='green', s=50)
    plt.scatter(list3X,list3Y,marker='^',c='blue', s=50)
#     #plotting centroids
    plt.scatter(cent1X,cent1Y,marker='o',c='red', s=100)
    plt.scatter(cent2X,cent2Y,marker='o',c='green', s=100)
    plt.scatter(cent3X,cent3Y,marker='o',c='blue', s=100)
    
def euclideanDistance(point1x,point1y,point2x,point2y):
    distance = ((point2x-point1x)*(point2x-point1x))+((point2y-point1y)*(point2y-point1y))
    return np.sqrt(abs(distance))

def clustering(inputSample,noOfClusters,centroid,x,y,z):
    
    inputXcoOrdinate = inputSample[:,0]
    inputYcoOrdinate = inputSample[:,1]
    centroXcoOrdinate = centroid[:,0]
    centridYcoOrdinate = centroid[:,1]
    classVector = []
    
    for i in range(0,len(inputXcoOrdinate)):
        distance = []
        for j in range(0,len(centroXcoOrdinate)):
            #print(inputXcoOrdinate[i],inputYcoOrdinate[i],centroXcoOrdinate[j],centridYcoOrdinate[j])
            dist = euclideanDistance(inputXcoOrdinate[i],inputYcoOrdinate[i],centroXcoOrdinate[j],centridYcoOrdinate[j])
            distance.append(dist)
        minDistance = min(distance)
        #print(distance)
        for k in range(0,len(distance)):
            if minDistance == distance[k]:
                classVector.append(k+1)
                
    #plotting the points
    list1X = []
    list1Y = []
    list2X = []
    list2Y =[]
    list3X = []
    list3Y = []
    for i in range(0,len(inputXcoOrdinate)):
        if classVector[i] == 1:
            list1X.append(inputXcoOrdinate[i])
            list1Y.append(inputYcoOrdinate[i])
        elif classVector[i] == 2:
            list2X.append(inputXcoOrdinate[i])
            list2Y.append(inputYcoOrdinate[i])
        elif classVector[i] == 3:
            list3X.append(inputXcoOrdinate[i])
            list3Y.append(inputYcoOrdinate[i])
            
    cent1X = centroXcoOrdinate[0]
    cent1Y = centridYcoOrdinate[0]
    
    cent2X = centroXcoOrdinate[1]
    cent2Y = centridYcoOrdinate[1]
    
    cent3X = centroXcoOrdinate[2]
    cent3Y = centridYcoOrdinate[2]
   
    #plotting sample points
    plt.subplot(x,y,z)
    plottingGraph(list1X,list1Y,list2X,list2Y,list3X,list3Y,cent1X,cent1Y,cent2X,cent2Y,cent3X,cent3Y)
   
    #plt.text(list1X[0],list1Y[0],'x')
    
    #compute updated x
    X1,Y1 = computeMean(list1X,list1Y)
    X2,Y2 = computeMean(list2X,list2Y)
    X3,Y3 = computeMean(list3X,list3Y)
    
    M1 = [X1,Y1]
    M2 = [X2,Y2]
    M3 = [X3,Y3]
    
    #potting the updated means
    
    plt.subplot(x,y,z+1)
    plottingGraph(list1X,list1Y,list2X,list2Y,list3X,list3Y,X1,Y1,X2,Y2,X3,Y3)
    
    return classVector,M1,M2,M3

def euclideanDistance3Cord(point1,point2):
    point1x = point1[0]
    point1y = point1[1]
    point1z = point1[2]
    
    point2x = point2[0]
    point2y = point2[1]
    point2z = point2[2]
    
    distance = ((point2x-point1x)*(point2x-point1x))+((point2y-point1y)*(point2y-point1y))+((point2z-point1z)*(point2z-point1z))
    
    return np.sqrt(distance)


def imageQuantization(image,k,Centroids):
    ClusterImage = np.zeros((image.shape[1],image.shape[0]))
    #listOfPixels = []
    for i in range(0,image.shape[0]):
        for j in range(0,image.shape[1]):
            distance = []
            for k in range(0,len(Centroids)):
                #print("test")
                temp = euclideanDistance3Cord(image[i,j],Centroids[k])
                distance.append(temp)
            ClusterNumber = distance.index(min(distance))
            ClusterImage[i,j] = int(ClusterNumber+1)
            
    return ClusterImage
   
def Centroid3D(list1):
    if len(list1) !=0:
        CentroX = list1[:,0]
        CentroY = list1[:,1]
        CentroZ = list1[:,2]

        sumX = 0
        sumY = 0
        sumZ = 0
        for i in range(0,len(CentroX)):
            sumX = CentroX[i] + sumX
            sumY = CentroY[i] + sumY
            sumZ = CentroZ[i] + sumZ

        sumX = sumX / len(CentroX)
        sumY = sumY / len(CentroY)
        sumZ = sumZ / len(CentroZ)

        return sumX,sumY,sumZ
    else:
        return 0,0,0
 


# In[113]:


def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip


# In[114]:


def guassianDistributionCal(inputSample,mean,covarinace):
    y = multivariate_normal.pdf(inputSample,mean,covarinace)
    return y
    
def Centroid3D(list1):
    if len(list1) !=0:
        CentroX = list1[:,0]
        CentroY = list1[:,1]

        sumX = 0
        sumY = 0
        #sumZ = 0
        for i in range(0,len(CentroX)):
            sumX = CentroX[i] + sumX
            sumY = CentroY[i] + sumY
            #sumZ = CentroZ[i] + sumZ

        sumX = sumX / len(CentroX)
        sumY = sumY / len(CentroY)
        #sumZ = sumZ / len(CentroZ)

        return sumX,sumY
    else:
        return 0,0,0    
    
    
def covarinceCal(mean,inputSamples):
    inputSamples = np.array(inputSamples)
    
    inputX = inputSamples[:,0]
    inputY = inputSamples[:,1]
    
    value1 = 0
    for i in range(0,len(inputX)):
        value1 = value1 + ((inputX[i]-mean[0])*(inputX[i]-mean[0]))
    
    value1 = value1 / len(inputX)
    
    value2 = 0
    for i in range(0,len(inputX)):
        value2 = value2 + ((inputX[i]-mean[0])*(inputY[i]-mean[1]))
    value2 = value2 / len(inputX)
    
    value3 = 0
    for i in range(0,len(inputX)):
        value3 = value3 + ((inputY[i]-mean[1])*(inputY[i]-mean[1]))
    value3 = value3 / len(inputX)
    
    covarinceMat = np.array([[value1,value2],[value2,value3]])
    
    return covarinceMat


    
        
def MeanCal(probabiityList1,SampleList1):
    sum1OfProb = 0
    newProb = []
    for i in range(0,len(probabiityList1)):
        #print(probabiityList1[i])
        #print(SampleList1[i])
        newProb.append([np.dot(probabiityList1[i],SampleList1[i])])
        sum1OfProb = sum1OfProb + probabiityList1[i]

    #print(newProb[0])
    newProb = np.array(newProb)
    #print(newProb)
    #print(newProb)
    #print(newProb)
    newProb = newProb[:,0]

    #print(newProb)
    newProbX = newProb[:,0]
    #print(newProbX)
    newProbY = newProb[:,1]



    sumXnum = np.sum(newProbX)
    sumYnum = np.sum(newProbY)

    sumXnum = sumXnum / sum1OfProb
    sumYnum = sumYnum / sum1OfProb

    #print(sumXnum)
    #print(sumYnum)
    return sumXnum,sumYnum
    
def newMean(probabiityList1,SampleList1):
    SampleList1 = np.array(SampleList1)
    inputX = SampleList1[:,0]
    inputY = SampleList1[:,1]
    
    Mean = np.zeros((1,2))
    for i in range(0,len(probabiityList1)):
        dataSample = np.array([[inputX[i],inputY[i]]])
        Mean = np.add(Mean,np.dot(probabiityList1[i],dataSample))
    Mean = np.dot(1/np.sum(probabiityList1),Mean)
    
    return Mean

def covarinceCalci(probabiityList1,SampleList1,Mean):
    SampleList1 = np.array(SampleList1)
    Mean = np.array(Mean)
    inputX = SampleList1[:,0]
    inputY = SampleList1[:,1]
    sum1OfProb = 0
    newProb = []
    covarinceMat = np.zeros((2,2))
    #print(covarinceMat)
    for i in range(0,len(probabiityList1)):
        
        dataSample = np.array([[inputX[i],inputY[i]]])
        MeanValue = np.array([[Mean[0],Mean[1]]])
        
        #print(dataSample)
        #print(MeanValue)
        subtarctValue = np.subtract(dataSample,MeanValue)
        #print(subtarctValue)
        
        
        subValueTransp = np.transpose(subtarctValue)
        #print(subValueTransp.shape)
        mulValue = np.multiply(subValueTransp,subtarctValue)
        covarinceMat = np.add(covarinceMat,(np.dot(probabiityList1[i],mulValue)))
        #print(covarinceMat)
    covarinceMat = np.dot(1/np.sum(probabiityList1),covarinceMat)
        
    #print(covarinceMat)
        
    return covarinceMat

def covCal(probabiityList1,inputSamples,mean):
    
    inputSamples = np.array(inputSamples)

    inputX = inputSamples[:,0]
    inputY = inputSamples[:,1]

    value1 = 0
    print(mean[0])
    for i in range(0,len(inputX)):
        value1 = value1 + ((inputX[i]-mean[0])*(inputX[i]-mean[0])*probabiityList1[i])
    print("value1" + str(value1))
    value1 = value1 / np.sum(probabiityList1)

    value2 = 0
    for i in range(0,len(inputX)):
        value2 = value2 + ((inputX[i]-mean[0])*(inputY[i]-mean[1])*probabiityList1[i])
    print("value1" + str(value2))
    value2 = value2 / np.sum(probabiityList1)

    value3 = 0
    for i in range(0,len(inputX)):
        value3 = value3 + ((inputY[i]-mean[1])*(inputY[i]-mean[1])*probabiityList1[i])
    print("value1" + str(value3))
    value3 = value3 / np.sum(probabiityList1)

    covarinceMat = np.array([[value1,value2],[value2,value3]])

    return covarinceMat
    
        
    

def GMMAlgorithm(inputSamples,Mean,Covarinace,iterationCount,y):
    z =1
    Cluster1 = []
    Cluster2 = []
    Cluster3 = []
    Meanllist = []
    covarinceList = []
    samplelist = []
    
    for m in range(0,iterationCount):
        ClusterNumber = []
        probabiityList1 = []
        SampleList1 = []
        probabiityList2 = []
        SampleList2 = []
        probabiityList3 = []
        SampleList3 = []
        cluster1prob = []
        cluster2prob = []
        cluster3prob = []
        for i in range(0,len(inputSamples)):
            probability = []
            for j in range(0,len(Mean)):

                temp = guassianDistributionCal(inputSamples[i],Mean[j],Covarinace[j]) 
                if j == 0:
                    #print("1")
                    cluster1prob.append(temp)
                elif j == 1:
                    #print("2")
                    cluster2prob.append(temp)
                elif j == 2:
                    #print("3")
                    cluster3prob.append(temp)
                probability.append(temp)
            #print(probability)
            ClusterValue = probability.index(max(probability))
            ClusterNumber.append(ClusterValue+1)
            if ClusterValue == 0:
                probabiityList1.append(probability[ClusterValue])
                SampleList1.append(inputSamples[i])
            elif ClusterValue == 1:
                probabiityList2.append(probability[ClusterValue])
                SampleList2.append(inputSamples[i])
            elif ClusterValue == 2:
                probabiityList3.append(probability[ClusterValue])
                SampleList3.append(inputSamples[i])
                
        samplelsitConcat = [SampleList1,SampleList2,SampleList3]
        samplelist.append(samplelsitConcat)
                
        x1 = 1/3
        x2 = 1/3
        x3 = 1/3
        
        for i in range(0,len(probabiityList1)):
            probabiityList1[i] = probabiityList1[i] *x1
            
        for i in range(0,len(probabiityList2)):
            probabiityList2[i] = probabiityList2[i] *x2
           
        for i in range(0,len(probabiityList3)):
            probabiityList3[i] = probabiityList3[i] *x3

        #updating the gaussian mixture coefficients
        x1 = np.sum(probabiityList1) / len(probabiityList1)
        x2 = np.sum(probabiityList2) / len(probabiityList2)
        x3 = np.sum(probabiityList3) / len(probabiityList3)
        
        
        
        M1 = np.array(MeanCal(probabiityList1,SampleList1))
        M2 = np.array(MeanCal(probabiityList2,SampleList2))
        M3 = np.array(MeanCal(probabiityList3,SampleList3))
        
        probabiityList1 = np.array(probabiityList1)
        probabiityList2 = np.array(probabiityList2)
        probabiityList3 = np.array(probabiityList3)
        
        
        C1 = covarinceCalci(probabiityList1,SampleList1,M1)
        C2 = covarinceCalci(probabiityList2,SampleList2,M2)
        C3 = covarinceCalci(probabiityList3,SampleList3,M3)

        #updating the Mean for each iteration
        Mean = [M1,M2,M3]
        Meanllist.append(Mean)
        SampleList1 = np.array(SampleList1)
        SampleList2= np.array(SampleList2)
        SampleList3 = np.array(SampleList3)
        Cluster1 = SampleList1
        Cluster2 = SampleList2
        Cluster3 = SampleList3
        
        
        #updting the covarince
        Covarinace = [C1,C2,C3]
        covarinceList.append(Covarinace)
    if y!=1: 
        plt.scatter(SampleList1[:,0],SampleList1[:,1],marker='o',c='red', s=20)
        plt.scatter(SampleList2[:,0],SampleList2[:,1],marker='o',c='green', s=20)
        plt.scatter(SampleList3[:,0],SampleList3[:,1],marker='o',c='blue', s=20)
    #     #plotting centroids
        plt.scatter(M1[0],M1[1],marker='o',c='black', s=20)
        plt.scatter(M2[0],M2[1],marker='o',c='black', s=20)
        plt.scatter(M3[0],M3[1],marker='o',c='black', s=20)
        plot_cov_ellipse(Covarinace[0],Mean[0], nstd=4, alpha=0.5, color='red')
        plot_cov_ellipse(Covarinace[1],Mean[1], nstd=4, alpha=0.5, color='green')
        plot_cov_ellipse(Covarinace[2],Mean[2], nstd=4, alpha=0.5, color='blue')
        plt.savefig("/Users/vidyach/Desktop/cvip/project2/temp/task3_gmm_iter"+str(iterationCount) +".jpg")
        
    
    return Mean


# # GMM for (10,2) datasamples

# In[115]:


inputSample = np.array([[5.9,3.2],[4.6,2.9],[6.2,2.8],[4.7,3.2],[5.5,4.2],[5.0,3.0],[4.9,3.1],[6.7,3.1],[5.1,3.8],[6.0,3.0]])
#print(inputSample)
Means = np.array([[6.2,3.2],[6.6,3.7],[6.5,3.0]])
#print(Means[:,0])
covariance = np.array([[[0.5,0],[0,0.5]],[[0.5,0],[0,0.5]],[[0.5,0],[0,0.5]]])
#print(covariance[0])
updateMeans = GMMAlgorithm(inputSample,Means,covariance,1,1)
print("M1 : " +str(updateMeans[0]))
print("M2 : " +str(updateMeans[1]))
print("M3 : " +str(updateMeans[2]))
#print(updateMeans)
#print(updateMeans)


# # faithful dataset GMM 

# In[116]:



# please specify the k value below which is for the ith iteration
k = 5
Means = np.array([[4.0, 81],[2.0,57],[4.0,71]])

covariance = np.array([[[1.30,13.98],[13.98,184.82]],[[1.30,13.98],[13.98,184.82]],[[1.30,13.98],[13.98,184.82]]])

updateMeans = GMMAlgorithm(inputSample1,Means,covariance,k,2)


