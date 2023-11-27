data generated from D:\03github\cdpr-gnn-calibration\exp_data_processing\script_gen_cdpr_data_HIT.m

- no noise
- varying orientation

xlim_200_800_ylim_200_800： workspace in x,y \in [200,800]
xlim_100_900_ylim_100_900： workspace in x,y \in [100,900]
Note: 从measured traj. 可以发现，x,y_measured in [200,800]

c4_euler_zlim-15  15deg：z轴的euler(i.e., theta)范围 \in [-15,15] deg
c4_euler_zlim-25  25deg：z轴的euler(i.e., theta)范围 \in [-25,25] deg
Note: 从measured traj. 可以发现，theta_measured in [-12,12] deg

根据real exp setup在仿真器上搭建相应Config:ExpC4，之后随机采集100 traj. with 100 pt/traj.
 