*_cdprconf.csv
一行表示一条cable, 前3列Ai, 后3列Bi; Ai表示cable一端在framework上坐标(global frame)，Bi表示cable一端在End-Effector上坐标(local frame)
*_qlList.csv
一行表示一个(q-l), 前6列为q = (x,y,z,alpha,beta,gamma), 后m列表示l = (m个cable lengths)
*_sgset.csv
一行表示一个(start-goal)，前3列为轨迹起点坐标，后3列为轨迹终点坐标(皆为global frame)