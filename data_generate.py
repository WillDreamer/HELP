import math
import random
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
import numpy as np
mpl.rcParams['font.sans-serif'] = ['SimHei']  
# from concorde.tsp import TSPSolver


def calFitness(line,dis_matrix):
    dis_sum = 0
    dis = 0
    for i in range(len(line)-1):
        dis = dis_matrix.loc[line[i],line[i+1]]
        dis_sum = dis_sum+dis
    dis = dis_matrix.loc[line[-1],line[0]]
    dis_sum = dis_sum+dis
    
    return round(dis_sum,1)


def intialize(CityCoordinates,antNum):

    cityList,cityTabu = [None]*antNum,[None]*antNum
    for i in range(len(cityList)):
        city = random.randint(0, len(CityCoordinates)-1)
        cityList[i] = [city]
        cityTabu[i] = list(range(len(CityCoordinates)))
        cityTabu[i].remove(city)
    
    return cityList,cityTabu


def select(antCityList,antCityTabu,trans_p):

    while len(antCityTabu) > 0:  
        if len(antCityTabu) == 1:
            nextCity = antCityTabu[0]
        else:
            fitness = []
            for i in antCityTabu:fitness.append(trans_p.loc[antCityList[-1],i])
            sumFitness = sum(fitness)
            randNum = random.uniform(0, sumFitness)
            accumulator = 0.0
            for i, ele in enumerate(fitness):
                accumulator += ele
                if accumulator >= randNum:
                    nextCity = antCityTabu[i]
                    break
        antCityList.append(nextCity)
        antCityTabu.remove(nextCity)
    
    return antCityList


def calTrans_p(pheromone,alpha,beta,dis_matrix,Q):

    transProb = Q/dis_matrix 
    for i in range(len(transProb)):
        for j in range(len(transProb)):
            transProb.iloc[i,j] = pow(pheromone.iloc[i,j], alpha) * pow(transProb.iloc[i,j], beta)
    
    return transProb


def updatePheromone(pheromone,fit,antCity,rho,Q):

    for i in range(len(antCity)-1):
        pheromone.iloc[antCity[i],antCity[i+1]] += Q/fit
    pheromone.iloc[antCity[-1],antCity[0]] += Q/fit
    
    return pheromone

def draw_path(line,CityCoordinates,antNum,CityNum,iterMax,dataid):
    x,y= [],[]
    for i in line:
        Coordinate = CityCoordinates[i]
        x.append(Coordinate[0])
        y.append(Coordinate[1])
    x.append(x[0])
    y.append(y[0])
    
    plt.plot(x, y,'r-', color='#4169E1', alpha=0.8, linewidth=0.8)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('./data/Antnum{}_Citynum{}_{}Iterdata_{}.png'.format(antNum,CityNum,iterMax,dataid))
    plt.close()
    


if __name__ == '__main__':
    
    data_len = 10
    
    for dataid in range(data_len):
    
        CityNum = 100
        MinCoordinate = 0
        MaxCoordinate = 101
        iterMax = 80
        iterI = 0   
        antNum = 50 
        alpha = 2   
        beta = 1    
        rho = 0.2   
        Q = 100.0   

        best_fit = math.pow(10,10)
        best_line = []

        CityCoordinates = [(random.randint(MinCoordinate,MaxCoordinate),random.randint(MinCoordinate,MaxCoordinate)) for i in range(CityNum)]

        dis_matrix = pd.DataFrame(data=None,columns=range(len(CityCoordinates)),index=range(len(CityCoordinates)))
        for i in range(len(CityCoordinates)):
            xi,yi = CityCoordinates[i][0],CityCoordinates[i][1]
            for j in range(len(CityCoordinates)):
                xj,yj = CityCoordinates[j][0],CityCoordinates[j][1]
                if (xi==xj) & (yi==yj):
                    dis_matrix.iloc[i,j] = round(math.pow(10,10))
                else:
                    dis_matrix.iloc[i,j] = round(math.sqrt((xi-xj)**2+(yi-yj)**2),2)

        pheromone = pd.DataFrame(data=Q,columns=range(len(CityCoordinates)),index=range(len(CityCoordinates)))
        trans_p = calTrans_p(pheromone,alpha,beta,dis_matrix,Q)

        data_saved = np.zeros((iterMax,antNum,CityNum,CityNum))
        route = np.zeros((iterMax,antNum,CityNum))
        target_saved = np.zeros((iterMax,antNum,1))

        while iterI < iterMax:
   
            antCityList,antCityTabu = intialize(CityCoordinates,antNum) 
            fitList = [None]*antNum

            for i in range(antNum): 
                antCityList[i] = select(antCityList[i],antCityTabu[i],trans_p)
                fitList[i] = calFitness(antCityList[i],dis_matrix)     
                pheromone = updatePheromone(pheromone,fitList[i],antCityList[i],rho,Q)
                trans_p = calTrans_p(pheromone,alpha,beta,dis_matrix,Q)
                data_saved[iterI,i,:,:]=pheromone


            for j in range(len(antCityList)):
                for idx, k in enumerate(antCityList[j]):
                    route[iterI,j,idx] = k
                    
            for tidx,t in enumerate(fitList):
                target_saved[iterI,tidx,:] = t 

            if best_fit >= min(fitList):
                best_fit = min(fitList)
                best_line = antCityList[fitList.index(min(fitList))]

            print("Iter:",iterI+1, "with the shortest distance:", best_fit)
            iterI += 1
            pheromone = pheromone*(1-rho)

        print(best_line)
        draw_path(best_line,CityCoordinates,antNum,CityNum,iterMax,dataid)
        dicts = {'city':dis_matrix,'pheromone':data_saved,'route':route,'target':target_saved}
        np.save('./data/Antnum{}_Citynum{}_{}Iterdata_{}'.format(antNum,CityNum,iterMax,dataid),dicts)