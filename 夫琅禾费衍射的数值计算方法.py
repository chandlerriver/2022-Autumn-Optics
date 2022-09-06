import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.integrate import dblquad
import math
import datetime
import multiprocessing as mp
import time

n = 100

def final_fun(name, param):
    result = []
    
    X0 = 0.4
    Y0 = 0.4
    
    global n
    k = 0.6
    f = 1 

    u = "1"
    x1 = 10
    x2 = -10
    y1 = lambda x: 10
    y2 = lambda x: -10
        
    A = np.linspace(-X0,X0,n)
    B = np.linspace(-Y0,Y0,n)
    
    for params in param:
        x = A[params[0]]
        y = B[params[1]]

        Aa = math.asin((x**2/(f**2+x**2+y**2))**0.5)
        Bb = math.asin((y**2/(f**2+x**2+y**2))**0.5)
        
        U1 = dblquad(lambda x,y:eval(u)*math.cos(2*math.pi*(x*math.sin(Aa)+y*math.sin(Bb))/k)
                     ,x1,x2,y1,y2)
        U2 = dblquad(lambda x,y:eval(u)*math.sin(2*math.pi*(x*math.sin(Aa)+y*math.sin(Bb))/k)
                     ,x1,x2,y1,y2)
        U = U1[0]**2 + U2[0]**2
        result.append(U)
    return {name:result}


if __name__ == '__main__': 
    start_time = time.time()
    num_cores = int(mp.cpu_count())
    pool = mp.Pool(num_cores)
    param_dict = {"task1":[[0,i] for i in range(n)],
"task2":[[1,i] for i in range(n)],
"task3":[[2,i] for i in range(n)],
"task4":[[3,i] for i in range(n)],
"task5":[[4,i] for i in range(n)],
"task6":[[5,i] for i in range(n)],
"task7":[[6,i] for i in range(n)],
"task8":[[7,i] for i in range(n)],
"task9":[[8,i] for i in range(n)],
"task10":[[9,i] for i in range(n)],
"task11":[[10,i] for i in range(n)],
"task12":[[11,i] for i in range(n)],
"task13":[[12,i] for i in range(n)],
"task14":[[13,i] for i in range(n)],
"task15":[[14,i] for i in range(n)],
"task16":[[15,i] for i in range(n)],
"task17":[[16,i] for i in range(n)],
"task18":[[17,i] for i in range(n)],
"task19":[[18,i] for i in range(n)],
"task20":[[19,i] for i in range(n)],
"task21":[[20,i] for i in range(n)],
"task22":[[21,i] for i in range(n)],
"task23":[[22,i] for i in range(n)],
"task24":[[23,i] for i in range(n)],
"task25":[[24,i] for i in range(n)],
"task26":[[25,i] for i in range(n)],
"task27":[[26,i] for i in range(n)],
"task28":[[27,i] for i in range(n)],
"task29":[[28,i] for i in range(n)],
"task30":[[29,i] for i in range(n)],
"task31":[[30,i] for i in range(n)],
"task32":[[31,i] for i in range(n)],
"task33":[[32,i] for i in range(n)],
"task34":[[33,i] for i in range(n)],
"task35":[[34,i] for i in range(n)],
"task36":[[35,i] for i in range(n)],
"task37":[[36,i] for i in range(n)],
"task38":[[37,i] for i in range(n)],
"task39":[[38,i] for i in range(n)],
"task40":[[39,i] for i in range(n)],
"task41":[[40,i] for i in range(n)],
"task42":[[41,i] for i in range(n)],
"task43":[[42,i] for i in range(n)],
"task44":[[43,i] for i in range(n)],
"task45":[[44,i] for i in range(n)],
"task46":[[45,i] for i in range(n)],
"task47":[[46,i] for i in range(n)],
"task48":[[47,i] for i in range(n)],
"task49":[[48,i] for i in range(n)],
"task50":[[49,i] for i in range(n)],
"task51":[[50,i] for i in range(n)],
"task52":[[51,i] for i in range(n)],
"task53":[[52,i] for i in range(n)],
"task54":[[53,i] for i in range(n)],
"task55":[[54,i] for i in range(n)],
"task56":[[55,i] for i in range(n)],
"task57":[[56,i] for i in range(n)],
"task58":[[57,i] for i in range(n)],
"task59":[[58,i] for i in range(n)],
"task60":[[59,i] for i in range(n)],
"task61":[[60,i] for i in range(n)],
"task62":[[61,i] for i in range(n)],
"task63":[[62,i] for i in range(n)],
"task64":[[63,i] for i in range(n)],
"task65":[[64,i] for i in range(n)],
"task66":[[65,i] for i in range(n)],
"task67":[[66,i] for i in range(n)],
"task68":[[67,i] for i in range(n)],
"task69":[[68,i] for i in range(n)],
"task70":[[69,i] for i in range(n)],
"task71":[[70,i] for i in range(n)],
"task72":[[71,i] for i in range(n)],
"task73":[[72,i] for i in range(n)],
"task74":[[73,i] for i in range(n)],
"task75":[[74,i] for i in range(n)],
"task76":[[75,i] for i in range(n)],
"task77":[[76,i] for i in range(n)],
"task78":[[77,i] for i in range(n)],
"task79":[[78,i] for i in range(n)],
"task80":[[79,i] for i in range(n)],
"task81":[[80,i] for i in range(n)],
"task82":[[81,i] for i in range(n)],
"task83":[[82,i] for i in range(n)],
"task84":[[83,i] for i in range(n)],
"task85":[[84,i] for i in range(n)],
"task86":[[85,i] for i in range(n)],
"task87":[[86,i] for i in range(n)],
"task88":[[87,i] for i in range(n)],
"task89":[[88,i] for i in range(n)],
"task90":[[89,i] for i in range(n)],
"task91":[[90,i] for i in range(n)],
"task92":[[91,i] for i in range(n)],
"task93":[[92,i] for i in range(n)],
"task94":[[93,i] for i in range(n)],
"task95":[[94,i] for i in range(n)],
"task96":[[95,i] for i in range(n)],
"task97":[[96,i] for i in range(n)],
"task98":[[97,i] for i in range(n)],
"task99":[[98,i] for i in range(n)],
"task100":[[99,i] for i in range(n)]}
    
    results = [pool.apply_async(final_fun, args=(name, param)) for name, param in param_dict.items()]
    results = [p.get() for p in results]
    end_time = time.time()
    use_time = end_time - start_time
    print("多进程计算 共消耗: " + "{:.2f}".format(use_time) + " 秒")
    data = [i["task"+str(results.index(i)+1)] for i in results]
    data = np.array(data)
    f = open("diffraction.txt","w")
    f.write(str(data))
    f.close()
    fig,ax = plt.subplots(figsize=(11,12))
    sns.heatmap(data, ax=ax,vmin=0,vmax=10,cmap='YlOrRd',\
        linewidths=2,cbar=False)


    plt.title('diffraction') 
    plt.ylabel('y_label')  
    plt.xlabel('x_label')   
    plt.savefig("diffraction.jpg")
    plt.show()
