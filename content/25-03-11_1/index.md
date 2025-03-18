---
emoji: 🏋🏻
title: "[백준] 14502번 연구소 (Python 파이썬)"
date: '2025-03-11 00:00:00'
author: 변우중
tags: 백준 14502 python 파이썬
categories: Algorithm
---
출처 : https://www.acmicpc.net/problem/14502

백준 14502번 연구소

&nbsp;

## 문제

---

인체에 치명적인 바이러스를 연구하던 연구소에서 바이러스가 유출되었다. 다행히 바이러스는 아직 퍼지지 않았고, 바이러스의 확산을 막기 위해서 연구소에 벽을 세우려고 한다.

연구소는 크기가 N×M인 직사각형으로 나타낼 수 있으며, 직사각형은 1×1 크기의 정사각형으로 나누어져 있다. 연구소는 빈 칸, 벽으로 이루어져 있으며, 벽은 칸 하나를 가득 차지한다.

일부 칸은 바이러스가 존재하며, 이 바이러스는 상하좌우로 인접한 빈 칸으로 모두 퍼져나갈 수 있다. 새로 세울 수 있는 벽의 개수는 3개이며, 꼭 3개를 세워야 한다.

예를 들어, 아래와 같이 연구소가 생긴 경우를 살펴보자.

```
2 0 0 0 1 1 0
0 0 1 0 1 2 0
0 1 1 0 1 0 0
0 1 0 0 0 0 0
0 0 0 0 0 1 1
0 1 0 0 0 0 0
0 1 0 0 0 0 0
```

이때, 0은 빈 칸, 1은 벽, 2는 바이러스가 있는 곳이다. 아무런 벽을 세우지 않는다면, 바이러스는 모든 빈 칸으로 퍼져나갈 수 있다.

2행 1열, 1행 2열, 4행 6열에 벽을 세운다면 지도의 모양은 아래와 같아지게 된다.

```
2 1 0 0 1 1 0
1 0 1 0 1 2 0
0 1 1 0 1 0 0
0 1 0 0 0 1 0
0 0 0 0 0 1 1
0 1 0 0 0 0 0
0 1 0 0 0 0 0
```

바이러스가 퍼진 뒤의 모습은 아래와 같아진다.

```
2 1 0 0 1 1 2
1 0 1 0 1 2 2
0 1 1 0 1 2 2
0 1 0 0 0 1 2
0 0 0 0 0 1 1
0 1 0 0 0 0 0
0 1 0 0 0 0 0
```

벽을 3개 세운 뒤, 바이러스가 퍼질 수 없는 곳을 안전 영역이라고 한다. 위의 지도에서 안전 영역의 크기는 27이다.

연구소의 지도가 주어졌을 때 얻을 수 있는 안전 영역 크기의 최댓값을 구하는 프로그램을 작성하시오.

&nbsp;

## 문제 파악

---

### 1. 생각의 흐름 💡

* 바이러스가 있는 2의 위치에서 상하좌우로 0이 있는 곳을 탐색하여 2로 변경해야 한다.
  => BFS 이용하자.
* 0의 위치에 벽(1)을 3개를 세워 바이러스의 경로를 차단할 수 있다.
  * *현재 바이러스가 있는 곳에서 상하좌우를 탐색하여 감염시키는 것을 1개의 턴이라고 했을 때,*
    *매 턴마다 몇 개의 감염이 늘어나는지 파악하여 **3개가 늘어나는 턴에서 벽을 세우면 될까?***
    => 이전의 턴에서 벽으로 막을 수 있으면 막는 것이 더 낫기 때문에 **잘못된 접근**임
  * 각각의 바이러스가 감염이 늘어나는 개수가 너무 무작위기 때문에 **현재 턴에서 감염이 늘어나는 개수와**
    **이전 턴에서 감염이 늘어나는 개수들을 비교해서 벽을 막는 것은 너무 복잡함**
* 그러면 **벽을 세우는 모든 경우에서 가장 감염이 적게 되는 경우를 찾아서** 그 경우에서 빈 칸(0)의 개수를 출력하자.
  => 계산복잡도가 높아질 수 있지만 가장 최선의 방법으로 보임

&nbsp;

### 2. 문제 해결에 필요한 것 🗝️

* 3개의 벽을 세우는 모든 조합을 찾아야 함 (itertools의 combinations() 함수 이용)
* **3개의 벽을 세운 후에 BFS 진행**하여 빈 칸(0)의 개수를 set에 저장함
  * list 대신 set을 이용해 조금이라도 복잡도를 낮춤 (어차피 우리는 빈 칸 개수의 최댓값만 필요함!!)
  * 벽을 세우는 함수(wall())를 먼저 호출한 후, 벽을 세우는 함수에서 벽을 세우는 각 경우마다 BFS 함수(bfs_empty_count())를 호출함
  * BFS 함수로 현재 벽을 세우는 조합에서 빈 칸의 개수를 반환함
* 벽을 세우는 각 경우마다 queue를 새롭게 생성해야 하고, 연구소 감염 맵(lab)도 deep copy한 것으로 진행함

&nbsp;

## 문제 해결

---

### 1. 소스 코드

``````python
from collections import deque
from itertools import combinations
import copy

# 모든 조합의 벽 세우기
def wall(empty_count):
    walls_combi = list(combinations(empty_loc, 3))
  
    # 벽을 세우는 조합 하나씩 접근
    for walls in walls_combi:
        new_lab = copy.deepcopy(lab) # 연구소 맵 깊은 복사
        new_queue = deque(queue) # 새로운 큐 생성
  
        # 현재 조합인 3개의 벽을 세움
        new_lab[walls[0][0]][walls[0][1]] = 1
        new_lab[walls[1][0]][walls[1][1]] = 1
        new_lab[walls[2][0]][walls[2][1]] = 1
  
        # 벽 세운 후의 lab에서 bfs 실행하여 빈 칸 개수를 set에 추가
        empty_count.add(bfs_empty_count(new_lab, new_queue))

    # 빈 칸(0)인 곳의 개수들 중에서 최댓값 반환
    return max(empty_count)

def bfs_empty_count(nlab, nqueue):  
    while nqueue:
        x, y = nqueue.popleft()
        for i in range(4):
            # 0일 때, 2로 바꾸고 큐에 추가
            if (0 <= x + dx[i] < m) and (0 <= y + dy[i] < n):
                if nlab[x + dx[i]][y + dy[i]] == 0:
                    nlab[x + dx[i]][y + dy[i]] = 2
                    nqueue.append([x + dx[i], y + dy[i]])

    # 빈 칸(0)인 곳의 개수
    n_empty_count = sum(row.count(0) for row in nlab)
  
    return n_empty_count


m, n = map(int, input().split())
lab = [list(map(int, input().split())) for _ in range(m)]
dx = [1, -1, 0, 0]
dy = [0, 0, 1, -1]
queue = deque()
empty_loc = list()
empty_count = set()


for i in range(m):
    for j in range(n):
        # 2의 위치(바이러스 위치) 저장
        if lab[i][j] == 2:
            queue.append([i, j])
        # 0의 위치(빈 칸 위치) 저장
        if lab[i][j] == 0:
            empty_loc.append((i, j))
  

print (wall(empty_count))
``````

### 2. 소스 코드 해석

``````python
m, n = map(int, input().split())
lab = [list(map(int, input().split())) for _ in range(m)]
dx = [1, -1, 0, 0]
dy = [0, 0, 1, -1]
queue = deque()
empty_loc = list()
empty_count = set()


for i in range(m):
    for j in range(n):
        # 2의 위치(바이러스 위치) 저장
        if lab[i][j] == 2:
            queue.append([i, j])
        # 0의 위치(빈 칸 위치) 저장
        if lab[i][j] == 0:
            empty_loc.append((i, j))
  

print (wall(empty_count))
``````

* 행의 수(m), 열의 수(n), 연구소 감염 맵(lab), dx, dy, queue, 0의 위치(empty_loc), 벽을 세우는 경우마다 빈 칸(0)의 개수를 저장하는 set 을 정의한다.
* 2의 위치(초기 바이러스 위치)와 0의 위치(빈 칸)를 각각 queue와 empty_loc에 저장한다.
* empty_count(벽을 세우는 경우마다 빈 칸의 개수 집합)를 인자로 하여 wall() 함수를 호출한다.

&nbsp;

**[wall() 함수]**

``````python
def wall(empty_count):
    walls_combi = list(combinations(empty_loc, 3))
  
    # 벽을 세우는 조합 하나씩 접근
    for walls in walls_combi:
        new_lab = copy.deepcopy(lab) # 연구소 맵 깊은 복사
        new_queue = deque(queue) # 새로운 큐 생성
  
        # 현재 조합인 3개의 벽을 세움
        new_lab[walls[0][0]][walls[0][1]] = 1
        new_lab[walls[1][0]][walls[1][1]] = 1
        new_lab[walls[2][0]][walls[2][1]] = 1
  
        # 벽 세운 후의 lab에서 bfs 실행하여 빈 칸 개수를 set에 추가
        empty_count.add(bfs_empty_count(new_lab, new_queue))

    # 빈 칸(0)인 곳의 개수들 중에서 최댓값 반환
    return max(empty_count)
``````

* wall() 함수에서 3개의 벽의 모든 조합을 생성한다.
* 각 벽의 조합을 for문을 통해 접근한다.
  * 각 벽의 조합에서 연구소 맵을 깊은 복사를 진행하고(new_lab), 이미 만들어져 있는 queue로 새로운 큐(nqueue)를 생성한다.
  * 현재 조합인 3개의 벽을 세운다. (new_lab이 변형됨)
  * bfs_empty_count() 함수를 **현재 연구소 맵인 new_lab**과 **2의 위치를 저장한 new_queue**를 인자로 하여 호출한다.
    => 호출 결과는 해당 벽의 조합에서의 빈 칸 개수임
* 빈 칸(0)의 개수들 중에 최댓값을 반환한다.

&nbsp;

**[bfs_empty_count() 함수]**

```python
def bfs_empty_count(nlab, nqueue):  
    while nqueue:
        x, y = nqueue.popleft()
        for i in range(4):
            # 0일 때, 2로 바꾸고 큐에 추가
            if (0 <= x + dx[i] < m) and (0 <= y + dy[i] < n):
                if nlab[x + dx[i]][y + dy[i]] == 0:
                    nlab[x + dx[i]][y + dy[i]] = 2
                    nqueue.append([x + dx[i], y + dy[i]])

    # 빈 칸(0)인 곳의 개수
    n_empty_count = sum(row.count(0) for row in nlab)
  
    return n_empty_count
```

* 인자로 받은 "**현재**" 연구소 맵(nlab)과 2의 위치를 저장한 큐(nqueue)에 대해 코드가 진행된다.
* **이미 벽을 세운 상태에서 바이러스 감염을 진행됐을 때 빈 칸이 얼마나 남을지를 보는 것**이다!!
* 큐(nqueue)가 빌 때까지 while문을 진행한다.
  * nqueue에서 바이러스 위치 하나를 뽑아온다.
  * 현재 뽑아온 바이러스 위치에서 dx와 dy를 이용해 상하좌우를 for문을 이용해 탐색한다.
    * 연구소 맵을 벗어나지 않는 공간에서 상하좌우를 이동했을 때, 그 값이 0이면 2로 바꾼 후에 nqueue에 추가한다.
  * 감염된 위치에서 이를 반복할 것임!!
* 큐가 비었으면 감염이 다 진행된 것이므로, 이제 빈 칸(0)의 개수를 세어 반환하여 함수 종료한다.
  => 그러면, wall() 함수에서 다음 3개의 벽의 조합에서 BFS를 또 진행할 것이다!!
  => 모든 조합에서 BFS 진행하면 wall() 함수에서 빈 칸(0) 개수의 최댓값을 반환할 것이다.

끝. (어려웠다.)

```toc

```
