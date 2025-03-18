---
emoji: 🏋🏻
title: "[백준] 7576번 토마토 (Python 파이썬)"
date: '2025-03-12 00:00:00'
author: 변우중
tags: 백준 7576 python 파이썬
categories: Algorithm
---
출처 : https://www.acmicpc.net/problem/7576

백준 7576번 토마토

&nbsp;

## 문제

---

철수의 토마토 농장에서는 토마토를 보관하는 큰 창고를 가지고 있다. 토마토는 아래의 그림과 같이 격자 모양 상자의 칸에 하나씩 넣어서 창고에 보관한다.

![img](https://u.acmicpc.net/de29c64f-dee7-4fe0-afa9-afd6fc4aad3a/Screen%20Shot%202021-06-22%20at%202.41.22%20PM.png)

창고에 보관되는 토마토들 중에는 잘 익은 것도 있지만, 아직 익지 않은 토마토들도 있을 수 있다. 보관 후 하루가 지나면, 익은 토마토들의 인접한 곳에 있는 익지 않은 토마토들은 익은 토마토의 영향을 받아 익게 된다. 하나의 토마토의 인접한 곳은 왼쪽, 오른쪽, 앞, 뒤 네 방향에 있는 토마토를 의미한다. 대각선 방향에 있는 토마토들에게는 영향을 주지 못하며, 토마토가 혼자 저절로 익는 경우는 없다고 가정한다. 철수는 창고에 보관된 토마토들이 며칠이 지나면 다 익게 되는지, 그 최소 일수를 알고 싶어 한다.

토마토를 창고에 보관하는 격자모양의 상자들의 크기와 익은 토마토들과 익지 않은 토마토들의 정보가 주어졌을 때, 며칠이 지나면 토마토들이 모두 익는지, 그 최소 일수를 구하는 프로그램을 작성하라. 단, 상자의 일부 칸에는 토마토가 들어있지 않을 수도 있다.

## 문제 파악

---

### 1. 생각의 흐름 💡

* bfs에서는 queue에서 뽑은 원소(익은 토마토 위치) 1개를 하나의 턴에 상하좌우를 탐색한다. 다시 말해, **상하좌우를 탐색하는 것이 모두 같은 날(day)에 이루어진다**.
* 여기서 깊이 생각해볼 문제가 있다.

  * 하루가 지나면서 3개의 토마토가 익었다고 가정하자. (queue에 3개 추가)
  * 그러면 ***그 익은 토마토 3개의 각 위치에서 다시 상하좌우를 탐색하여 queue에 추가하는 모든 위치를 같은 날로 인식하게 해야 하는데 어떤 식으로 할 것인가?***
    => queue에 익은 토마토의 위치를 추가할 때, **해당 토마토가 익은 날(day)을 함께 추가**해주자.
  * 해당 토마토가 익은 날을 모두 set()에 넣고 모든 토마토가 익은 날을 정리하고, 최댓값을 반환해주면 될 것 같다.
* queue가 빌 때까지 bfs를 진행했는데도(토마토를 끝까지 익혀도) 0이 남아 있으면 반환값을 -1로 해주면 될 것 같다.

&nbsp;

### 2. 문제 해결에 필요한 것 🗝️

* 익은 토마토의 위치와 익은 날짜를 함께 저장하는 queue가 필요할 것
* **현재 익은 토마토의 익은 날을 저장**하고, **상하좌우를 탐색**하여 **아직 익지 않은 토마토(0)의 위치와 다음 날을 queue에 저장**하도록 하는 함수가 필요할 것임 (**ripe() 함수**)
* **queue에서 원소 하나씩 뽑아서 위치와 익은 날을 ripe() 함수에 전달**해주는 함수 필요할 것임 (**bfs() 함수**)
  (최종적으로 반환할 값도 정해주는 함수임)

&nbsp;

## 문제 해결

---

### 1. 소스 코드

``````python
from collections import deque


def bfs():
    while queue:
        # queue에서 하나 뽑아옴 (각 원소는 위치와 queue에 추가된 날이 들어있음)
        x, y, day = queue.popleft()
  
        # queue가 빌 때까지, queue의 각 원소가 익은 날을 day_set에 계속 업데이트 함
        ripe(x, y, day)

    # 모두 익지 못하는 상황에 -1 출력 (bfs 끝까지 진행했는데도 0이 있는 상황)
    for i, row in enumerate(tomato_map):
        # 해당 row에 0이 하나라도 있으면 -1로 반환
        if 0 in row:
            return -1
  
    return max(day_set)


def ripe(rx, ry, rday):
    # 현재 queue의 원소가 익은 날을 day_set에 저장함
    day_set.add(rday)
  
    for i in range(4):
        if (0 <= rx + dx[i] < m) and (0 <= ry + dy[i] < n):
            # 익지 않은 토마토가 상하좌우에 있을 때
            if tomato_map[rx + dx[i]][ry + dy[i]] == 0:
                # 상하좌우의 토마토가 익음
                tomato_map[rx + dx[i]][ry + dy[i]] = 1

                # 다음 queue의 원소를 다음 날로 설정해 함께 저장
                # rday += 1로 쓰면, 한 원소에 대해 상하좌우를 탐색할 때 같은 rday가 아니게 됨
                queue.append([rx + dx[i], ry + dy[i], rday + 1])



n, m = map(int, input().split())
tomato_map = [list(map(int, input().split())) for _ in range(m)]
queue = deque()
dx = [1, -1, 0, 0]
dy = [0, 0, 1, -1]
day_set = set()

for i in range(m):
    for j in range(n):
        if tomato_map[i][j] == 1:
            queue.append([i, j, 0]) # 초기 1의 위치, 0일차
            day_set.add(0)


print (bfs())
``````

&nbsp;

### 2. 소스 코드 해석

``````python
n, m = map(int, input().split())
tomato_map = [list(map(int, input().split())) for _ in range(m)]
queue = deque()
dx = [1, -1, 0, 0]
dy = [0, 0, 1, -1]
day_set = set()

for i in range(m):
    for j in range(n):
        if tomato_map[i][j] == 1:
            queue.append([i, j, 0]) # 초기 1의 위치, 0일차
            day_set.add(0)


print (bfs())
``````

* 일반적인 bfs 문제 초기 설정들이다.
* **queue에 초기 익은 토마토(1)의 위치와 0(0일차)을 함께 묶어 넣어준다** !!
  (day_set에도 0을 추가해준다.)

&nbsp;

**[bfs() 함수]**

```python
def bfs():
    while queue:
        # queue에서 하나 뽑아옴 (각 원소는 위치와 queue에 추가된 날이 들어있음)
        x, y, day = queue.popleft()
  
        # queue가 빌 때까지, queue의 각 원소가 익은 날을 day_set에 계속 업데이트 함
        ripe(x, y, day)

    # 모두 익지 못하는 상황에 -1 출력 (bfs 끝까지 진행했는데도 0이 있는 상황)
    for i, row in enumerate(tomato_map):
        # 해당 row에 0이 하나라도 있으면 -1로 반환
        if 0 in row:
            return -1
  
    return max(day_set)
```

* queue에서 익은 토마토의 위치 x, y와 익은 날짜 day를 하나씩 함께 뽑아온다.
* x, y, day를 인자로 하여 ripe() 함수를 호출한다.

  * 상하좌우 탐색하고 익지 않은 토마토들을 queue에 담아오고 (위치와 다음 일차를 담음)
  * 토마토가 익은 날들을 day_set에 담는다.
* queue가 빌 때까지 (최대한 토마토를 다 익게 한 후) 진행한 후에 for문을 통해 아직 익지 않은 토마토(0)가 있으면 -1을 반환하도록 한다.
* 0이 없으면 day_set의 최댓값을 반환하도록 한다.

&nbsp;

**[ripe() 함수]**

``````python
def ripe(rx, ry, rday):
    # 현재 queue의 원소가 익은 날을 day_set에 저장함
    day_set.add(rday)
  
    for i in range(4):
        if (0 <= rx + dx[i] < m) and (0 <= ry + dy[i] < n):
            # 익지 않은 토마토가 상하좌우에 있을 때
            if tomato_map[rx + dx[i]][ry + dy[i]] == 0:
                # 상하좌우의 토마토가 익음
                tomato_map[rx + dx[i]][ry + dy[i]] = 1

                # 다음 queue의 원소를 다음 날로 설정해 함께 저장
                # rday += 1로 쓰면, 한 원소에 대해 상하좌우를 탐색할 때 같은 rday가 아니게 됨
                queue.append([rx + dx[i], ry + dy[i], rday + 1])
``````

* 현재 queue의 원소(익은 토마토 위치와 익은 날)를 인자로 받아서 day_set에 먼저 익은 날을 추가해준다.
* for문을 이용해 현재 익은 토마토의 상하좌우를 탐색하고, 익지 않은 토마토가 있을 때 익은 토마토(1)로 바꿔주고 queue에 **해당 토마토의 위치**와 **다음 날(rday+1)**을 함께 추가해주자.
* 아, 그리고...

  ```python
  if (0 <= rx + dx[i] < m) and (0 <= ry + dy[i] < n):
  ```

  잊지 말자......

&nbsp;

끝. 이제 bfs 문제 풀 만 한 것 같은데... 자만하지 말고 더 달리자!!!

```toc

```
