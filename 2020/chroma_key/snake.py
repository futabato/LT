import numpy as np
import cv2
import math
import copy


#画像の読み込み
test = cv2.imread("./image/summerwars.jpg", cv2.IMREAD_COLOR)#BGRなので気をつける

gray_test = cv2.imread("./image/summerwars.jpg",cv2.IMREAD_GRAYSCALE)
height = test.shape[0]
width = test.shape[1] 

N = 2000

v = np.zeros((N,2))
start_v = np.zeros((N,2))
vec_g = np.zeros(2)
for i in range(0,N):
    if(i<N/4):
        v[i] = [height/(N/4)*i,0]
    elif(i<2*N/4):
        v[i] = [height-1,width/(N/4)*(i-N/4)]
    elif(i<3*N/4):
        v[i] = [height-1-height/(N/4)*(i-2*N/4),width-1]
    else:
        v[i] = [0,width-1 - width/(N/4)*(i-3*N/4)]

start_v = copy.deepcopy(v)
display = copy.deepcopy(test)

#パラメータ
alpha =1
beta = 1
gamma = 10
kappa =1

def EpsIn(vec0,vec1,vec2):#test
    value = 0
    value += alpha*np.linalg.norm(vec1-vec0)**2+beta*np.linalg.norm(vec2-2*vec1+vec0)**2
    value /= 2
#     print("In:"+str(value))
    return value

def EpsEx(vec0,pix):#gray
    value = 0
    x = int(vec0[0])
    y = int(vec0[1])

    if(x+1 >= height or y+1 >= width):
        return float('inf') 
    else:
        I = [abs(int(pix[x+1,y]) - int(pix[x,y])) ,abs(int(pix[x,y+1])-int(pix[x,y]))]
        value = -gamma*np.linalg.norm(I)**2
#         print("Ex:"+str(value))
        return value

def EpsCon(vec0,vec_g):#test
    value = 0
    value += kappa*np.linalg.norm((vec0[0] - vec_g[0],vec0[1]-vec_g[1]))**2
#     print("Con:"+str(value))
    return value

def Energy(vec0,vec1,vec2,vec_g,pix):
    value = 0
    value = EpsIn(vec0,vec1,vec2)+EpsEx(vec0,pix)+EpsCon(vec0,vec_g)
#     print("Result:"+str(value))
    return value
#探索
n = 500
dx = [-1, -1, -1, 0, 0, 0, 1, 1, 1]
dy = [1, 0, -1, 1, 0, -1, 1, 0, -1]
# dx = [1,1,1,0,0,0,-1,-1,-1]
# dy = [1,-1,0,1,-1,0,1,-1,0]
#210
#543
#876

flag = 4
for loop in range(0,n):
    for i in range(0,N):
        flag = 4
        eps_min = float('inf') 
        vec_g = [0,0]

        for j in range(0,9):            
            move  = [v[i,0]+dx[j], v[i,1]+dy[j]]
            if(move[0] < 0 or move[1] < 0 or move[0] >= height  or move[1] >= width):
                continue #はみ出し処理

            #画像中心を基準に
            vec_g = [int(height/2),int(width/2)]

            energy = Energy(move,v[(i+1)%N],v[(i+2)%N],vec_g,gray_test)
            if(eps_min>energy):
                eps_min = energy
                flag = j
        v[i] += [dx[flag],dy[flag]]

        #逐次書き出し
    if(loop%100==0):
        cv2.imwrite('./image/result'+str(loop)+'.jpg', display)
        display = copy.deepcopy(test)
        for i in range(0,N):
            cv2.line(display, (int(v[i,1]),int(v[i,0])), (int(v[(i+1)%N,1]),int(v[(i+1)%N,0])), (0, 255, 0), 2)



for i in range(0,N):
    cv2.line(display, (int(v[i,1]),int(v[i,0])), (int(v[(i+1)%N,1]),int(v[(i+1)%N,0])), (0, 255, 0), 2)

for i in range(0,N):
    cv2.line(display, (int(start_v[i,1]),int(start_v[i,0])), (int(start_v[(i+1)%N,1]),int(start_v[(i+1)%N,0])), (255, 0, 0), 2)

cv2.imwrite('./image/result.png', display)