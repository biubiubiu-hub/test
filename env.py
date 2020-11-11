import numpy as np
import time
import matplotlib.pyplot as plt
class env(object):
    def __init__(self, UE_num, env_radius, max_mov_distance_per):
        self.UE_num = UE_num
        self.env_radius = env_radius
        self.UE_max_mov_dis_per = max_mov_distance_per
        self.UAV_pos = np.array([0, 0, 0])
        self.UAV_pos_buffer = np.zeros([1, 3])  # save the uav location in history
        self.timslot = 100 * 10**(-3)
        self.UE_tx_power = 16      #unit is dbW
        self.UAV_rx_gain = 20      #db
        self.UE_band = 180 * 10**3 #     180K
        self.rx_power_stack = np.zeros([1,self.UE_num])
        self.s = np.zeros([1,UE_num])     #state
        self.noise_PSD = -204 # -174 dbm/Hz

        self.com_traffic =[]
        self.succeed_trans_UE=[]   #完成传输的用户
        self.uav_speed = 300
        self.state_dim = self.UE_num * 2    #状态空间的大小
        self.action_dim = 1             #动作空间的大小
        self.action_bound = 180         #动作范围

    def reset(self):
        UE_pos = np.random.uniform(-self.env_radius, self.env_radius, [self.UE_num * 10, 2])
        UE_index = np.where(np.sum(UE_pos ** 2,axis=1) <= self.env_radius ** 2)
        UE_pos = UE_pos[UE_index][0]
        self.UE_pos = np.append(UE_pos, 0)
        #---------------------------------
        self.UAV_pos = [-self.env_radius, 0, 3]
        self.UAV_pos_buffer = np.array(self.UAV_pos)
        self.UAV_pos_buffer = np.vstack((self.UAV_pos_buffer,self.UAV_pos))
        s = self.UE_pos - self.UAV_pos
        s =np.delete(s,2)    #去除高度信息
        s1 = []
        for i in s:
            s1 = np.hstack((s1, i.tolist()))
        s = s1
        return s

    def calc_rate(self,dis):
        shadowing_var = 1
        path_loss = 145.4 + 37.5 * np.log10(dis)
        chan_loss = path_loss + np.random.normal(0, shadowing_var)
        rx_power = 10 ** ((self.UE_tx_power - chan_loss + self.UAV_rx_gain) / 10)
        rate = self.UE_band * np.log10(
            1 + rx_power / (10 ** (self.noise_PSD / 10) * self.UE_band))
        com_traffic = rate * self.timslot  # 单个时隙内的通信流量
        return com_traffic

    def step(self,a):
        # self.UE_random_mov() #UE随机移动一点距离
        # 信道模型
        UE_pos = self.UE_pos
        shadowing_var = 1  # rayleigh fading shadowing variance 8dB
        ##--------------------------------------------------------------------
        dis = np.sqrt(np.sum((self.UAV_pos - UE_pos) ** 2))
        ##-------------------------------------------------------------------##
        self.path_loss = 145.4 + 37.5 * np.log10(dis).reshape(-1, 1)
        self.chan_loss = self.path_loss + np.random.normal(0, shadowing_var, self.UE_num).reshape(-1, 1)
        self.rx_power = 10 ** ((self.UE_tx_power - self.chan_loss + self.UAV_rx_gain) / 10)
        self.rx_power = self.rx_power.reshape(1, -1)
        self.rx_power_stack = np.vstack((self.rx_power_stack, self.rx_power))
        printer = False
        if printer:
            print('dis is', dis)
            print('path loss is ', self.path_loss)
            print('rx_power is', self.rx_power)
            print('chan loss is',self.chan_loss)
        if self.rx_power_stack.shape[0] > 10:
            self.rx_power_stack = np.delete(self.rx_power_stack, 0, 0)
        # -----------------------------------------------------------------------
        rate = self.UE_band * np.log10(
            1 + self.rx_power / (10 ** (self.noise_PSD / 10) * self.UE_band))
        # -----------------------------------------------------------------------
        # print('rate is',rate)
        self.com_traffic = rate * self.timslot  # 单个时隙内的通信流量
        #-------------------------------------------------------
        self.env_action(a)
        r=self.com_traffic[0][0] * 10000
        s_,_ = self.env_state()
        done = 0
        if r >50000:
            done = 1
        return s_, r, done

    def env_state(self):
        #---------------------------------
        s = self.UE_pos - self.UAV_pos
        s = s[:2]
        done = 0
        return s,done

    def env_action(self,angle):
        v=self.uav_speed
        rad = angle/180 * np.pi
        xv = v* np.cos(rad)
        yv = v* np.sin(rad)
        zv=0
        mov_dis =[xv,yv,zv]
        mov_dis =np.array([xv,yv,zv]) * self.timslot
        UAV_pos = np.add(self.UAV_pos, mov_dis)
        if np.sum(UAV_pos ** 2) < self.env_radius **2 :
            self.UAV_pos = UAV_pos
        self.UAV_pos_buffer = np.vstack((self.UAV_pos_buffer,np.array(self.UAV_pos)))
        # print(self.UAV_pos_buffer)

    def env_reward(self):
        rate_threshold = 5
        # print('com_traffic is',self.com_traffic)
        # succeed_trans_index_inslot = np.where(self.com_traffic > rate_threshold)[1] #当前时隙满足传输要求的用户
        # # self.succeed_trans_UE = []
        # new = [i for i in succeed_trans_index_inslot if i not in self.succeed_trans_UE]
        # # print('new',new)
        # old = [i for i in succeed_trans_index_inslot if i in self.succeed_trans_UE]
        # # print('old',old)
        # # print('self.succeed',self.succeed_trans_UE)
        # self.succeed_trans_UE=np.hstack((self.succeed_trans_UE,new))
        # self.succeed_trans_UE.sort()
        # r = 10**len(new)-10*len(old)
        # if len(self.succeed_trans_UE) == self.UE_num:
        #     r += 100000
        return r

    def UE_random_mov(self):
        UE_mov_dis = np.random.uniform(-self.UE_max_mov_dis_per,self.UE_max_mov_dis_per,[self.UE_num,2])
        UE_mov_dis = np.append(UE_mov_dis,np.zeros([self.UE_num,1]),1)
        self.UE_pos = self.UE_pos + UE_mov_dis

    def render(self):
        pos_buffer =self.UAV_pos_buffer
        fig = plt.figure(0)
        ax = fig.add_subplot(111, projection='3d')
        #---------------------UAV---------------------------
        X = pos_buffer[:, 0]
        Y = pos_buffer[:, 1]
        Z = pos_buffer[:, 2]
        ax.scatter(X, Y, Z)
        #-------UE----------------------------
        X_u = self.UE_pos[:, 0]
        Y_u = self.UE_pos[:, 1]
        Z_u = 0
        ax.scatter(X_u,Y_u,Z_u, c='r')
        #---------env circle-----------------
        theta = np.arange(0, 2 * np.pi, 0.01)
        x = self.env_radius * np.cos(theta)
        y = self.env_radius * np.sin(theta)
        ax.scatter(x, y,1,c='g')
        lim_scale = 600
        plt.xlim((-lim_scale, lim_scale))
        plt.ylim((-lim_scale, lim_scale))
        ax.set_zlim(0, 30)
        # plt.zlim((-1000,1000))
        plt.show()

# env = env(UE_num=1,env_radius=800,max_mov_distance_per=1)
# s = env.reset()
# print('s init is',s)
# a = np.random.uniform(-180,180)
# s_,r,done = env.step(a)
# print('S_',s_)
# print('r',r)
# print('done',done)






