# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 22:24:05 2024

@author: user
"""

import numpy as np
import heapq # min heap 을 구현하는 heap queue
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

# 지도 1 벽, 0 : 빈공간
grid = np.array([
    [0,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
    [1,0,0,1,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
    [1,0,0,0,0,0,0,0,1,0,0,0,0,1,1,1,1,1,0,1],
    [1,0,0,0,0,0,0,0,1,0,0,0,0,1,1,1,0,0,0,0],
    [1,0,0,0,0,0,0,0,1,0,0,0,0,1,1,1,0,0,0,0],
    [0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
    [1,1,1,1,1,0,0,0,1,0,0,0,0,0,0,0,1,1,1,1],
    [0,0,0,0,0,0,0,0,1,0,0,1,1,1,1,0,1,1,1,0],
    [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,1,1,1,0,0],
    [0,0,1,1,1,1,1,1,1,0,0,0,0,1,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
    [0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])

start = (0,0)   #시작 위치
goal = (0,19) # 목적지 위치

#휴리스틱 함수 h() : a와  b사이의 유클리드 거리 
def heuristic(a,b):
    return np.sqrt((b[0]-a[0])**2+(b[1]-a[1])**2)

#A* 알고리즘
def Astar(array,start,goal):
    neighbors =[(0,1),(0,-1),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)] #이웃 위치
    close_set = set() #탐색이 종료된 위치들의 집합 
    came_from ={}
    gscore = {start:0} #시작 위치의 g()값
    fscore = {start:heuristic(start,goal)} #시작위치의 f() 값
    oheap = [] #min-heap
    heapq.heappush(oheap,(fscore[start],start)) #(거리,출발지)min-heap에 저장 
    
    while oheap:
        current = heapq.heappop(oheap)[1] # f()값이 최소인 노드 수출
        if current == goal: #목적지 도착
            data = []
            while current in came_from: #목적지에서 역순으로 경로 추출 
                data.append(current)
                current = came_from[current]
            return data
        close_set.add(current) #current 위치를 탐색이 종료된 것으로 간주
        
        for i,j in neighbors: #current 위치의 각 이웃 위치에 대해 f() 값 계산
            neighbor = current[0]+i,current[1]+j # 이웃 위치
            if 0 <= neighbor[0] < array.shape[0]:
                if 0 <= neighbor[1] < array.shape[1]:
                    if array[neighbor[0]][neighbor[1]] == 1: #벽
                        continue
                    else: # y방향의 경계를 벗어난 상황
                        continue
                else:  # x방향의 경계를 벗어난 상황
                    continue
                    
                temp_g_score = gscore[current]+heuristic(current,neighbor) # g^(n) = g(c)+h((c,n))
                if neighbor in close_set and temp_g_score >= gscore.get(neighbor,0):
                    continue #이미 방문한 위치이면서 g^() 값이 기존 g()값보다 큰 경우 --> 무시
                
                if temp_g_score < gscore.get(neighbor,0) or neighbor not in [i[1]for i in oheap]:
                    #g^(n)<g(n) 이거나, n을 처음 방문한 경우 
                    came_from[neighbor]=current #neighbor 에 도달한 최선의 경로에서 직선 위치는 current
                    gscore[neighbor] = temp_g_score #g^(n)=g(n)
                    fscore[neighbor] = temp_g_score + heuristic(neighbor,goal) # f()=g()+h()
                    heapq.heappush(oheap,(fscore[neighbor],neighbor)) # min heap 에 (f(),neighbor) 삽입
    return False
    
route = Astar(grid,start,goal)
route = route + [start] # 출발 위치 추가
route = route[::-1] #역순으로 변환
print('경로:', route)

#route 에서 x와 y 좌표 추출
x_coords = []
y_coords = []
for i in (range(0,len(route))):
    x = route[i][0]
    y = route[i][1]
    x_coords.append(x)
    y_coords.append(y)
    
 #지도와 경도 그리기
fig, ax = plt.subplots(figsize=(12,12))
ax.imshow(grid,cmap=plt.cm.Pastel1)
ax.scatter(start[1],start[0],marker="*",color="red",s=200)
ax.scatter(goal[1],goal[0],marker="*",color="green",s=200)
ax.plot(y_coords,x_coords,color="blue")
plt.show()