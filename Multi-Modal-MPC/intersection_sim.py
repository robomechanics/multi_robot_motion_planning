import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import matplotlib.transforms as tf
import pdb
from random import sample
import copy
import polytope as pc
from pedestrian_agent import Agent
from pytope import Polytope
from itertools import product
import random

class Simulator():
    def __init__(self,
                agents, 
                T_FINAL     = 1000,
                reduced_mode=False,
                viz_preds=True,
                eval_mode=False
                ):
        
        self.reduced_mode=reduced_mode
        self._make_lanes() 
        self.agents=agents

        self.N_TV=0
        self.tvs=[]
        self.peds=[]
        self.tv_idxs=[]
        self.ped_idxs =[]
        self.mm_preds=[]
        self.ev_sols =[]
        self.viz_preds=viz_preds
        self.eval_mode = eval_mode
        
        # Only 3 TVs
        self.N_modes=[3,3,3]

       

        for i,v in enumerate(self.agents):
            if v.role=="TV":
                self.N_TV+=1
                self.tvs.append(v)
                self.tv_idxs.append(i)
            elif v.role=="ped":
                self.peds.append(v)
                self.ped_idxs.append(i)
            else:
                self.ev=v
        self.t=0
        self.T=T_FINAL
    
    def _get_idm_params(self, v, cl, v_, cl_, verbose = False):
        '''
        Our IDM-based Interaction Engine
        '''
        v_des=self.routes[cl](v[0]+2.0)[-1]

        dv   =0.0 
        ds   =1e5

        psi=self.routes[cl](v[0])[2]
            
        for i, (vh, clh) in enumerate(zip(v_,cl_)):

            if clh in self.modes[self.sources[cl]]:

                
                vh_s=vh[0]
                vh_psi=self.routes[clh](vh[0])[2]

                if vh_s-v[0]>=0. and np.abs(np.cos(psi-vh_psi))>=0.3:
                    ds=max(vh_s-v[0]-6.0,0.01)
                    vh_psi=self.routes[clh](vh[0])[2]
                    dv=v[1]-vh[1]*np.cos(psi-vh_psi)+6.
                    if verbose:
                        print(f'Vehicle #{i} (EV: {i==len(cl_)-1}):=   Branch #1: Keep going')
                    break
            
            if self.sinks[cl]==self.sinks[clh]:
                
                vh_pos=self.routes[clh](vh[0]-.0)[:2].reshape((-1,1))
                vh_s=self._g2f(vh_pos,cl)
                v_s_on_vh = self._g2f(self.routes[cl](v[0]-.0)[:2].reshape((-1,1)),clh)
                vh_psi=self.routes[clh](vh[0])[2]

                if np.abs(np.sin(float(2*psi)))<=1e-3 and vh_s-v[0]>=0. and vh_s-v[0]<=30. and self._check_out_inter(cl,v[0]):
                    if (vh[1]>=1 and vh_s-v[0]<=18) and (1<=(v_s_on_vh - vh[0]) <= 25):
                        ds=0.01
                        dv=8
                        if verbose:
                            print(f'Vehicle #{i} (EV: {i==len(cl_)-1}):=   Branch #2: Yield')
                    else:
                        ds=max(vh_s-v[0]-6.,0.01) 
                        dv=v[1]-vh[1]*np.cos(psi-vh_psi)
                        dv+= 5. if 2*float(psi)%np.pi==0 else -5.

                        if verbose:
                            print(f'Vehicle #{i} (EV: {i==len(cl_)-1}):=   Branch #3: Keep Going. Vehicle ahead')
                            
                        
                        if vh[1]<0 and np.abs(np.sin(2*psi))>=0.001: #if vh is hestitating, just go
                            ds = 1e5
                            dv = 0.0

                elif vh_s-v[0]-6.0 < 0. and vh_s - v[0]>=.0 and np.abs(np.cos(psi-vh_psi))>=0.3:
                    ds=max(vh_s-v[0]-6.,0.01) 
                    dv=v[1]-vh[1]*np.cos(psi-vh_psi)
                    dv+= 5. if 2*float(psi)%np.pi==0 else -5.

                    if verbose:
                        print(f'Vehicle #{i} (EV: {i==len(cl_)-1}):=   Branch #4: Keep Going. Vehicle ahead')
                else:
                    if verbose:
                        print(f'Vehicle #{i} (EV: {i==len(cl_)-1}):=   Keep Going')
            else:

                vh_pos=self.routes[clh](vh[0]-.0)[:2].reshape((-1,1))
                vh_s=self._g2f(vh_pos,cl)
                vh_psi=self.routes[clh](vh[0]+0.)[2]

                p2p1=-self.routes[cl](v[0]+0.)[:2].reshape((-1,1))+vh_pos
                d1 = self.droutes[cl](v[0]+0.)[:2].reshape((-1,1))*20.      #30 m lookahead
                d2 = self.droutes[clh](vh[0]-.0)[:2].reshape((-1,1))*20.
                d1cd2, pcd2, pcd1  =ca.det(ca.horzcat(d1,d2)), ca.det(ca.horzcat(p2p1, d2)), ca.det(ca.horzcat(p2p1, d1))

                if int(d1cd2 > 0 or d1cd2 < 0)==1:
                    t, u = pcd2/d1cd2, pcd1/d1cd2
                    if bool(0.<=t<=1.) and bool(0.<=u<=1.):
                        if np.abs(np.sin(float(psi)))<=1e-3 and vh_s-v[0]>=0.5 and np.linalg.norm(p2p1)<=30. and self._check_out_inter(cl,v[0]):
                            # if vh[1]>=2.:
                            ds=0.01
                            dv=8.
                            if verbose:
                                print(f'Vehicle #{i} (EV: {i==len(cl_)-1}):=   Branch #5: Yield')
                            
                        elif vh_s-v[0]-6.0<0. and vh_s-v[0]>=0.5 and np.linalg.norm(p2p1)<=20. and not self._check_out_inter(clh,vh[0]):
                            ds=max(vh_s-v[0]-6.0,0.01)
                            dv=v[1]-vh[1]*np.cos(psi-vh_psi)
                            dv+=6. if np.abs(np.sin(float(psi)))<=1e-3 else 2.
                            if verbose:
                                    print(f'Vehicle #{i} (EV: {i==len(cl_)-1}):=   Branch #7: Keep Going. Vehicle ahead ')

                            if vh[1]<0 and np.abs(np.sin(psi))>=0.001: #if vh is hestitating, just go
                                ds = 1e5
                                dv = 0.0
                        else:
                            if verbose:
                                print(f'Vehicle #{i} (EV: {i==len(cl_)-1}):=   Branch #8: Keep Going')       

        if verbose:
            print(f'v_des, dv, ds: {v_des}, {dv}, {ds}')   
            print('++++++++++++++++++')      
        return v_des, dv, ds
            
            
    def done(self):
        #Reached the pre-defined time = limit of the simulator
        done = self.t==self.T or self.routes_pose[self.ev.cl][-1,-1]-self.ev.traj[0,self.ev.t]<=0.1

        if not done:
            self._TV_gen()
        else:
            print(f"EV reached {self.ev.traj[0,self.ev.t]}")

        return done
    
    def collision(self):
        #A collision occured during simulation
        return self._check_collision()
    
    def set_MPC_N(self, N):
        self.N=N
    
    def _TV_gen(self):
        #checks if TVs have reached destination and respawns TVs accordingly
        for i,v in enumerate(self.tvs):
            if self.routes_pose[v.cl][-1,-1]-v.traj[0,v.t]<=0.01:
                print("Resetting agent: {}".format(i+1))
                print("Reached {}".format(v.traj[0,v.t]))
                new_cl=sample(self.modes[self.sources[v.cl]],1)[0]


                if v.cl!=2 and v.cl!=4:
                    next_init_close=False
                    for vh in self.tvs:
                        if vh!=v and self.sources[v.cl]==self.sources[vh.cl] and np.abs(vh.traj[0,vh.t])<=6.:
                            next_init_close=True

                    init=np.array([-2.0,7.0]) if next_init_close else np.array([6.,6.0])
                else:
                    init=copy.copy(v.traj[:,v.t])
                v.reset_vehicle(init, new_cl)
    
    def _check_collision(self):
        ev_S=Polytope(self.ev.vB.A, self.ev.vB.b)
        psi=self.routes[self.ev.cl](self.ev.traj[0,self.ev.t])[-1]
        Rev=np.array([[np.cos(psi), -np.sin(psi)],[np.sin(psi), np.cos(psi)]]).squeeze()
        ev_S=Rev*ev_S+self.routes[self.ev.cl](self.ev.traj[0,self.ev.t])[:2]
        ev_S=pc.Polytope(ev_S.A, ev_S.b)
        for i, v in enumerate(self.tvs):
            tv_S=Polytope(v.vB.A, v.vB.b)
            psi=self.routes[v.cl](v.traj[0,v.t])[-1]
            Rtv=np.array([[np.cos(psi), -np.sin(psi)],[np.sin(psi), np.cos(psi)]]).squeeze()
            tv_S=Rtv*tv_S+self.routes[v.cl](v.traj[0,v.t])[:2]
            tv_S=pc.Polytope(tv_S.A, tv_S.b)
            tv_ev= tv_S.intersect(ev_S)
            if not pc.is_empty(tv_ev):
                print(f"EV Collided with TV {self.tv_idxs[i]} ({self.agents[self.tv_idxs[i]].cl}) at position {self.ev.traj[0, self.ev.t]}")
                return True
        
        return False



    def step(self, u_ev=None, verbose=False):

        for ind, v in enumerate(self.agents):
            if v != self.ev:
                v.traj_glob[:,v.t]=np.array(self.routes[v.cl](v.traj[0,v.t])[:3]).squeeze()
                if v.role!="ped":
                    if verbose:
                        print(f'TV{ind+1}')
                    idx_=set(self.tv_idxs)-set([ind])
                    v_ =[self.agents[k].traj[:,v.t] for k in idx_] + [self.ev.traj[:,v.t]]
                    cl_=[self.agents[k].cl for k in idx_] + [self.ev.cl]
                    v_des, dv, ds= self._get_idm_params(v.traj[:,v.t], v.cl, v_, cl_, verbose)
                    v.step(v.clip_vel_acc(v.traj[:,v.t],v.idm(v_des, dv, ds)))
                else:
                    v_des = self.routes[v.cl](.0+v.traj[0,v.t])[3]
                    v.step(v.idm(v_des))
                    
                    if np.abs(v.traj[0,v.t]-v.s_decision)<0.05:
                        if self.sources[v.cl]=="W":
                            v.cl = random.choice([5,6])
                        else:
                            v.cl = random.choice([8,9])

        self.ev.traj_glob[:,self.ev.t]=np.array(self.routes[self.ev.cl](self.ev.traj[0,self.ev.t])[:3]).squeeze()
        if not u_ev:
            v_ =[self.agents[k].traj[:,v.t] for k in self.tv_idxs]
            cl_=[v.cl for v in self.tvs]
            v_des, dv, ds= self._get_idm_params(self.ev.traj[:,self.ev.t], self.ev.cl, v_, cl_,verbose)
            self.ev.step(self.ev.clip_vel_acc(self.ev.traj[:,self.ev.t],self.ev.idm(v_des, dv, ds)))
        else:
            self.ev.step(u_ev)
        self.t+=1

       
    
    def get_update_dict(self, u_opt=None):

        z_lin, x_pos, dpos, mm_o_glob, mm_u_tvs, mm_routes, mm_droutes, mm_Qs =self._get_preds(u_opt)
        u_prev=self.ev.u[self.ev.t-1] if self.ev.t>0 else 0.

        update_dict={'x0': self.ev.traj[:,self.ev.t], 'u_prev': u_prev ,
                     'o0': [v.traj[:,v.t] for v in self.agents if v!=self.ev], 'o_glob': mm_o_glob, 
                     'routes': mm_routes, 'droutes': mm_droutes, 'Qs' : mm_Qs,
                     'z_lin': z_lin, 'x_pos':x_pos,  'dpos': dpos, 'u_tvs': mm_u_tvs }
        
        self.mm_preds.append(mm_o_glob)

        self.ev_sols.append(x_pos)

        return update_dict
    
   
    
    def _check_out_inter(self,cl,s):
        return (self.sources[cl]=="E" and (s <= self.routes_pose[2][-1,-1]+0.5))\
                 or (self.sources[cl]=="W" and (s <= 51.+0.5))
        
    def _get_preds(self, u_opt):
        '''
        Getting EV predictions from previous MPC solution.
        This is used for linearizing the collision avoidance constraints
        '''
        N=self.N
        
        
        x=self.ev.traj[:,self.ev.t].reshape((-1,1))+np.zeros((2,N+1))
    
        x_glob=self.routes[self.ev.cl](x[0,0])[:2].reshape((-1,1))+np.zeros((2, N+1))
        dx_glob=[ca.DM(2,1) for _ in range(N)]
        o=[v.traj[:,v.t].reshape((-1,1))+np.zeros((2,self.N+1)) for v in self.agents if v!=self.ev]
        o_glob=[self.routes[v.cl](v.traj[0,v.t])[:2].reshape((-1,1))+np.zeros((2,N+1)) for v in self.agents if v!=self.ev]
        u_tvs=[np.zeros((1,N)) for v in self.agents if v!=self.ev]
        tv_list  = self.tv_idxs
        ped_list = self.ped_idxs
        do_glob = [[ca.DM(2,1) for _ in range(N)] for v in self.agents if v!=self.ev]
        Qs = [[np.identity(2) for _ in range(N)] for v in self.agents if v!=self.ev]
        iSev=np.linalg.inv(self.ev.S)
        iSev[-1,-1]+=0.3
        Sev=np.linalg.inv(iSev)

        
        
        for t in range(N):
            if u_opt is None:
                v_ =[o[i][:,t] for i in tv_list]
                cl_=[v.cl for v in self.tvs]
                v_des, dv, ds= self._get_idm_params(x[:,t], self.ev.cl, v_, cl_)
                a=self.ev.clip_vel_acc(x[:,t], self.ev.idm(v_des, dv, ds))
                
            else:
                a=u_opt[t]
                
   
            x[:,t+1]=self.ev.get_next(x[:,t+1], a)
            x_glob[:,t+1]=self.routes[self.ev.cl](x[0,t+1])[:2]
            dx_glob[t]=self.droutes[self.ev.cl](x[0,t+1])[:2]
            psi= self.routes[self.ev.cl](x[0,t+1])[2]
            Rev=np.array([[np.cos(psi), np.sin(psi)],[-np.sin(psi), np.cos(psi)]]).squeeze().T
            
            for i in tv_list:
                idx_=set(tv_list)-set([i])
                v_ =[o[k][:,t] for k in idx_] + [x[:,t]]
   
                cl_=[self.agents[k].cl for k in idx_] + [self.ev.cl]

                v_des, dv, ds= self._get_idm_params(o[i][:,t], self.agents[i].cl, v_, cl_)
                a=self.agents[i].clip_vel_acc(o[i][:,t],self.agents[i].idm(v_des, dv, ds))
                u_tvs[i][0,t]=a
                o[i][:,t+1]=self.agents[i].get_next(o[i][:,t], a)
                o_glob[i][:,t+1]=self.routes[self.agents[i].cl](o[i][0,t+1])[:2]
                do_glob[i][t]=self.droutes[self.agents[i].cl](o[i][0,t+1])[:2]
                psi=self.routes[self.agents[i].cl](o[i][0,t+1])[2]
                Rtv=np.array([[np.cos(psi), np.sin(psi)],[-np.sin(psi), np.cos(psi)]]).squeeze().T

                mat=Rev@iSev@Rtv.T@self.agents[i].S@self.agents[i].S@Rtv@iSev@Rev.T
                E, V =np.linalg.eigh(mat)
                S=np.diag((E**(-0.5)+1.0)**(-2))
                Qs[i][t]=Sev@Rev.T@V@S@V.T@Rev@Sev if t <=4 else (1/5**2)*np.eye(2)
                
            for i in ped_list:
                v_des = self.routes[self.agents[i].cl](0.+o[i][0,t+1])[3]
                a=self.agents[i].idm(v_des)
                u_tvs[i][0,t]=a
                o[i][:,t+1]=self.agents[i].get_next(o[i][:,t], a)
                o_glob[i][:,t+1]=self.routes[self.agents[i].cl](o[i][0,t+1])[:2]
                do_glob[i][t]=self.droutes[self.agents[i].cl](o[i][0,t+1])[:2]
                psi=self.routes[self.agents[i].cl](o[i][0,t+1])[2]
                Rtv=np.array([[np.cos(psi), np.sin(psi)],[-np.sin(psi), np.cos(psi)]]).squeeze().T

                mat=Rev@iSev@Rtv.T@self.agents[i].S@self.agents[i].S@Rtv@iSev@Rev.T
                E, V =np.linalg.eigh(mat)
                S=np.diag((E**(-0.5)+1.0)**(-2))
                Qs[i][t]=Sev@Rev.T@V@S@V.T@Rev@Sev if t <=4 else (1/5**2)*np.eye(2)
                
        mm_o      = [[copy.deepcopy(o[i]) for _ in range(self.n_modes[i])] for i,v in enumerate(self.agents) if v!=self.ev]
        mm_o_glob = [[copy.deepcopy(o_glob[i]) for _ in range(self.n_modes[i])] for i,v in enumerate(self.agents) if v!=self.ev]
        mm_u_tvs  = [[copy.deepcopy(u_tvs[i]) for _ in range(self.n_modes[i])] for i,v in enumerate(self.agents) if v!=self.ev]
        mm_Qs     = [[copy.deepcopy(Qs[i]) for _ in range(self.n_modes[i])] for i,v in enumerate(self.agents) if v!=self.ev]
        mm_routes = [[copy.copy(self.routes[v.cl]) for _ in range(self.n_modes[i])] for i,v in enumerate(self.agents) if v!=self.ev]
        mm_droutes = [[copy.deepcopy(do_glob[i]) for _ in range(self.n_modes[i])] for i,v in enumerate(self.agents) if v!=self.ev]

        for i in tv_list:
            modes=set(self.modes[self.sources[self.agents[i].cl]])-set([self.agents[i].cl])
            
            for t in range(N):
                psi= self.routes[self.ev.cl](x[0,t+1])[2]
                Rev=np.array([[np.cos(psi), np.sin(psi)],[-np.sin(psi), np.cos(psi)]]).squeeze().T
                
                if  (self.sources[self.agents[i].cl]=="E" and self.agents[i].traj[0,self.t]<=self.routes_pose[2][-1,-1]+3.)\
                 or (self.sources[self.agents[i].cl]=="W" and self.agents[i].traj[0,self.t]<=51.+3.):
                    for j in modes:
                        n=self.modes[self.sources[self.agents[i].cl]].index(j)
                        
                        if t==0:
                            mm_routes[i][n]=self.routes[j]
                        idx_=set(tv_list)-set([i])
                        v_ =[o[k][:,t] for k in idx_]+[x[:,t]]
                        cl_=[self.agents[k].cl for k in idx_]+[self.ev.cl]

                        v_des, dv, ds= self._get_idm_params(mm_o[i][n][:,t], j, v_, cl_)
                        a=self.agents[i].clip_vel_acc(mm_o[i][n][:,t],self.agents[i].idm(v_des, dv, ds))

                        mm_o[i][n][:,t+1]=self.agents[i].get_next(mm_o[i][n][:,t], a)
                        mm_o_glob[i][n][:,t+1]=self.routes[j](mm_o[i][n][0,t+1])[:2]
                        mm_droutes[i][n][t]=self.droutes[j](mm_o[i][n][0,t+1])[:2]
                        mm_u_tvs[i][n][0,t]=a
                        psi=self.routes[j](mm_o[i][n][0,t+1])[2]
                        Rtv=np.array([[np.cos(psi), np.sin(psi)],[-np.sin(psi), np.cos(psi)]]).squeeze().T
                        mat=Rev@iSev@Rtv.T@self.agents[i].S@self.agents[i].S@Rtv@iSev@Rev.T 
                        E, V =np.linalg.eigh(mat)
                        S=np.diag((E**(-0.5)+1.0)**(-2))
                        mm_Qs[i][n][t]=Sev@Rev.T@V@S@V.T@Rev@Sev if t <=4 else (1/5**2)*np.eye(2)
                        
        for i in ped_list:
            modes=set(self.modes[self.sources[self.agents[i].cl]])-set([self.agents[i].cl])
            
            for t in range(N):
                psi= self.routes[self.ev.cl](x[0,t+1])[2]
                Rev=np.array([[np.cos(psi), np.sin(psi)],[-np.sin(psi), np.cos(psi)]]).squeeze().T
                
                for j in modes:
                    n=self.modes[self.sources[self.agents[i].cl]].index(j)
                    
                    if n+1== len(self.modes[self.sources[self.agents[i].cl]]) and self.agents[i].traj[0,self.t]>=1.0:
                        continue
                    else:
                    
                        if t==0:
                            mm_routes[i][n]=self.routes[j]
                        v_des = self.routes[j](.0+mm_o[i][n][0,t+1])[3]
                        a=self.agents[i].idm(v_des)
                        mm_o[i][n][:,t+1]=self.agents[i].get_next(mm_o[i][n][:,t], a)
                        mm_o_glob[i][n][:,t+1]=self.routes[j](mm_o[i][n][0,t+1])[:2]
                        mm_droutes[i][n][t]=self.droutes[j](mm_o[i][n][0,t+1])[:2]
                        mm_u_tvs[i][n][0,t]=a
                        psi=self.routes[j](mm_o[i][n][0,t+1])[2]
                        Rtv=np.array([[np.cos(psi), np.sin(psi)],[-np.sin(psi), np.cos(psi)]]).squeeze().T
                        mat=Rev@iSev@Rtv.T@self.agents[i].S@self.agents[i].S@Rtv@iSev@Rev.T 
                        E, V =np.linalg.eigh(mat)
                        S=np.diag((E**(-0.5)+1.0)**(-2))
                        mm_Qs[i][n][t]=Sev@Rev.T@V@S@V.T@Rev@Sev if t <=4 else (1/5**2)*np.eye(2)

        return x, x_glob, dx_glob, mm_o_glob, mm_u_tvs, mm_routes, mm_droutes, mm_Qs 
        
    
    def _make_lanes(self):
        #                         
        # lane numbering:= 0:W->E, 1:E->W, 2:E->W (slow), (straights)
        #                  3:W->N,                        (lefts)
        #                  4:E->N                         (rights) 
        self.modes   = {'E':[1,2,4,8, 9,10], 'W':[0,3,5,6,7]}
        self.sources = {0:'W', 1:'E', 2: 'E', 3: 'W', 4:'E', 5:'W', 6:'W', 7:'W', 8:'E', 9:'E', 10:'E'}
        self.sinks   = {0:'E', 1:'W', 2: 'W', 3:'N', 4:'N', 5:'E', 6:'E', 7:'E', 8:'W', 9:'W', 10:'W'}

        #               TV  ped_W  ped_E     
        self.n_modes  = [3,   3,    3 ]
       
        
        def _make_ca_fun(s, x, y, psi, v):

            x_ca= ca.interpolant("f2gx", "linear", [s], x)
            y_ca= ca.interpolant("f2gy", "linear", [s], y)
            psi_ca= ca.interpolant("f2gpsi", "linear", [s], psi)
            v_ca= ca.interpolant("f2gv", "linear", [s], v)
            s_sym=ca.MX.sym("s",1)

            glob_fun=ca.Function("fx",[s_sym], [ca.vertcat(x_ca(s_sym), y_ca(s_sym), psi_ca(s_sym), v_ca(s_sym))])

            return glob_fun
        
        def _make_jac_fun(pos_fun):
            s_sym=ca.MX.sym("s",1)
            pos_jac=ca.jacobian(pos_fun(s_sym), s_sym)
            return ca.Function("pos_jac",[s_sym], [pos_jac])

        self.routes_pose=[]
        self.droutes=[]

        straights_x=[]
        # 0 : W->E
        s=np.array([0,110])
        xs=np.array((-50,60))
        ys=np.array((0,0))
        psis=np.array((np.pi, np.pi))
        vs=np.array([8.5, 10.0])
        r_fun=_make_ca_fun(s,xs,ys, 0.*psis, vs)
        straights_x.append(r_fun)
        self.droutes.append(_make_jac_fun(r_fun))
        s_r=np.linspace(s[0], s[-1])
        r_p=np.array([r_fun(s_r[i])[:3] for i in range(s_r.shape[0])]).squeeze()
        self.routes_pose.append(np.vstack((r_p.T,s_r.reshape((1,-1)))))

        # 1 : E->W
        r_fun=_make_ca_fun(s,xs[::-1],ys+15., psis, vs)
        straights_x.append(r_fun)
        self.droutes.append(_make_jac_fun(r_fun))
        s_r=np.linspace(s[0], s[-1])
        r_p=np.array([r_fun(s_r[i])[:3] for i in range(s_r.shape[0])]).squeeze()
        self.routes_pose.append(np.vstack((r_p.T,s_r.reshape((1,-1)))))

        # 2 : E->W
        s=np.array([0,29])
        xs=np.array((60,31))
        ys=np.array((15,15))
        vs=np.array([9.5,0.0])
        r_fun=_make_ca_fun(s,xs,ys, -psis, vs)
        straights_x.append(r_fun)
        self.droutes.append(_make_jac_fun(r_fun))
        s_r=np.linspace(s[0], s[-1])
        r_p=np.array([r_fun(s_r[i])[:3] for i in range(s_r.shape[0])]).squeeze()
        self.routes_pose.append(np.vstack((r_p.T,s_r.reshape((1,-1)))))
        
        
        lefts_x=[]        
        # 3 : W->N
        thet=np.linspace(0,np.pi/2)

        x_f= lambda t :  8.5 + 15*np.sin(t)
        y_f= lambda t :  15 - 15*np.cos(t)
        s=np.hstack((np.array([0, 58.5]), 58.501 + 15.*thet, np.array([58.502+15.*np.pi/2, 58.5 + 15.*np.pi/2+25])))
        vs=np.hstack((np.array([10., 7.]), 6. + 0.*thet, np.array([6., 8.])))
        x_l=np.hstack((np.array([-50.,7.5]),x_f(thet), np.array([23.5, 23.5])))
        y_l=np.hstack((np.array([0.,0.]), y_f(thet), np.array([15, 40.])))

        psis=np.hstack((np.array([0.0, 0.]),  thet, 0.5*np.array([np.pi, np.pi])))
        r_fun=_make_ca_fun(s, x_l, y_l, psis, vs)
        lefts_x.append(r_fun)
        self.droutes.append(_make_jac_fun(r_fun))
        s_r=np.linspace(s[0], s[-1])
        r_p=np.array([r_fun(s_r[i])[:3] for i in range(s_r.shape[0])]).squeeze()
        self.routes_pose.append(np.vstack((r_p.T,s_r.reshape((1,-1)))))

        lefts = lefts_x

        rights=[] 

        # 4 : E->N
        x_f= lambda t :  31. - 7.5*np.sin(t)
        y_f= lambda t :  22.5 - 7.5*np.cos(t)
        s=np.hstack((np.array([0, 29.]), 29.001 + 7.5*thet, np.array([29.002+7.5*np.pi/2, 29.0 + 7.5*np.pi/2+17.5])))
        x_l=np.hstack((np.array([60.,31.]), x_f(thet), np.array([23.5, 23.5])))
        y_l=np.hstack((np.array([15.,15.]), y_f(thet), np.array([22.5, 40.])))
        psis=np.hstack((np.array([np.pi, np.pi]), np.pi-thet, 0.5*np.array([np.pi, np.pi])))
        vs=np.hstack((np.array([8, 4.8]), 4.8 + 0.*thet, np.array([4.8, 8])))
        r_fun=_make_ca_fun(s, x_l, y_l, psis,vs)
        rights.append(r_fun)
        self.droutes.append(_make_jac_fun(r_fun))
        s_r=np.linspace(s[0], s[-1])
        r_p=np.array([r_fun(s_r[i])[:3] for i in range(s_r.shape[0])]).squeeze()
        self.routes_pose.append(np.vstack((r_p.T,s_r.reshape((1,-1)))))
        
        ped_cross =[]
        # 5 : W->E (run)
        s=np.array([0,31])
        xs=np.array((1,32))
        ys=np.array((24,24))
        psis=np.array((np.pi, np.pi))
        vs=np.array([4.5, .0])
        r_fun=_make_ca_fun(s,xs,ys, 0.*psis, vs)
        ped_cross.append(r_fun)
        self.droutes.append(_make_jac_fun(r_fun))
        s_r=np.linspace(s[0], s[-1])
        r_p=np.array([r_fun(s_r[i])[:3] for i in range(s_r.shape[0])]).squeeze()
        self.routes_pose.append(np.vstack((r_p.T,s_r.reshape((1,-1)))))
        
        # 6 : W->E (walk)
        s=np.array([0,31])
        xs=np.array((1,32))
        ys=np.array((24,24))
        psis=np.array((np.pi, np.pi))
        vs=np.array([2., 0.])
        r_fun=_make_ca_fun(s,xs,ys, 0.*psis, vs)
        ped_cross.append(r_fun)
        self.droutes.append(_make_jac_fun(r_fun))
        s_r=np.linspace(s[0], s[-1])
        r_p=np.array([r_fun(s_r[i])[:3] for i in range(s_r.shape[0])]).squeeze()
        self.routes_pose.append(np.vstack((r_p.T,s_r.reshape((1,-1)))))
        
        # 7 : W->E (yield)
        s=np.array([0,1])
        xs=np.array((1,2))
        ys=np.array((24,24))
        psis=np.array((np.pi, np.pi))
        vs=np.array([0.5, 0])
        r_fun=_make_ca_fun(s,xs,ys, 0.*psis, vs)
        ped_cross.append(r_fun)
        self.droutes.append(_make_jac_fun(r_fun))
        s_r=np.linspace(s[0], s[-1])
        r_p=np.array([r_fun(s_r[i])[:3] for i in range(s_r.shape[0])]).squeeze()
        self.routes_pose.append(np.vstack((r_p.T,s_r.reshape((1,-1)))))
        

        # 8 : E->W (run)
        s=np.array([0,31])
        xs=np.array((1,32))
        ys=np.array((24,24))
        psis=np.array((np.pi, np.pi))
        vs=np.array([4.5, .0])
        r_fun=_make_ca_fun(s,xs[::-1],ys-1.0, psis, vs)
        ped_cross.append(r_fun)
        self.droutes.append(_make_jac_fun(r_fun))
        s_r=np.linspace(s[0], s[-1])
        r_p=np.array([r_fun(s_r[i])[:3] for i in range(s_r.shape[0])]).squeeze()
        self.routes_pose.append(np.vstack((r_p.T,s_r.reshape((1,-1)))))
        
        # 9 : E->W (walk)
        vs=np.array([2., 0.])
        r_fun=_make_ca_fun(s,xs[::-1],ys-1., psis, vs)
        ped_cross.append(r_fun)
        self.droutes.append(_make_jac_fun(r_fun))
        s_r=np.linspace(s[0], s[-1])
        r_p=np.array([r_fun(s_r[i])[:3] for i in range(s_r.shape[0])]).squeeze()
        self.routes_pose.append(np.vstack((r_p.T,s_r.reshape((1,-1)))))
        
        # 10 : E->W (yield)
        s=np.array([0,1])
        xs=np.array((31,32))
        vs=np.array([0.5, 0])
        r_fun=_make_ca_fun(s,xs[::-1],ys-1., psis, vs)
        ped_cross.append(r_fun)
        self.droutes.append(_make_jac_fun(r_fun))
        s_r=np.linspace(s[0], s[-1])
        r_p=np.array([r_fun(s_r[i])[:3] for i in range(s_r.shape[0])]).squeeze()
        self.routes_pose.append(np.vstack((r_p.T,s_r.reshape((1,-1)))))

        straights = straights_x
        
        self.routes=straights+lefts+rights+ped_cross

    def _g2f(self, pos, cl):
        idx=np.argmin(np.linalg.norm(self.routes_pose[cl][:2,:]-pos.reshape((-1,1)), axis=0))
        return self.routes_pose[cl][-1,idx]

    
    def draw_intersection(self, ax, i):

        _tf = lambda x: tf.Affine2D().rotate(x[-1]).translate(x[0], x[1])


        #Map Boundaries and roads
        ax.add_patch(Rectangle((-50, -7.5),110,30,linewidth=1,edgecolor='darkgrey', fc='darkgrey',fill=True, alpha=0.7))
        ax.add_patch(Rectangle((1, -7.5),30,47.5,linewidth=1,edgecolor='darkgrey', fc='darkgrey',fill=True, alpha=0.7))
        ax.plot([-50, 1], [22.5, 22.5], color='k', lw=2)
        ax.plot([-50, 60], [-7.5, -7.5], color='k', lw=2)
        ax.plot([31, 60], [22.5, 22.5], color='k', lw=2)
        ax.plot([1, 1], [22.5, 40], color='k', lw=2)
        ax.plot([31, 31], [22.5, 40], color='k', lw=2)
        
        for r in self.routes_pose[:5]:
            ax.plot(r[0,:], r[1,:], color='w', linewidth= 1.2, linestyle = (0, (5,10)))

        #Drawing agents
        v_color= {"EV": "g", "TV":"r","ped":"y"}
        v_alpha= {"EV": 1, "TV":0.5,"ped":1.}
        v_shapes=[]
        

        for k, v in enumerate(self.agents):
            
            v_pos=v.traj_glob[:,i]
   
            if v.role!="ped":
                v_shapes.append(Rectangle((0.-.3,0.-1.8),6.,3.6,linewidth=1., ec='k', fc=v_color[v.role], alpha = v_alpha[v.role]))
            else:
                
                v_shapes.append(Rectangle((0.-.5,0.-.5),1.,1.,linewidth=1., ec='k', fc=v_color[v.role], alpha = v_alpha[v.role]))
                
            v_shapes[-1].set_transform(_tf(v_pos)+ax.transData)
            ax.add_patch(v_shapes[-1])

                

        #Print EV States
        # ev_legend = Rectangle((-48,-15),6.,3.6,linewidth=1., ec='green', fc='green')
        # ax.add_patch(ev_legend)
        # ax.text(-40,-15,f's: {self.agents[-1].traj[0,i].round(1)}, vel: {self.agents[-1].traj[1,i].round(1)}')


        ax.set_xlim(-50,60)
        ax.set_ylim(-8,50)
        ax.axis('equal')
        
if __name__=="__main__":
    
    
    ev_noise_std=[0.001,0.01]
    ev=Agent(role='EV', cl=3, noise_std=ev_noise_std)
    tv_noise_std=[0.01, 0.1]
    agents=[Agent(role='TV', cl=2, state=np.array([0, 7.]), noise_std=tv_noise_std) for i in range(1)]
    agents.append(Agent(role='ped', cl=6, state=np.array([0., 2.5]), noise_std=tv_noise_std))
    agents.append(Agent(role='ped', cl=9, state=np.array([0., 2.5]), noise_std=tv_noise_std))

    tv_n_stds=[v.noise_std for v in agents]
    agents.append(ev)
    Sim=Simulator(agents)

    # smpc=SMPC_MMPreds(Sim.routes, ev, EV_NOISE_STD=ev_noise_std, TV_NOISE_STD=tv_n_stds)

    Sim.set_MPC_N(10)


    while Sim.t<250 and not Sim.done():
        # print("Time: ",Sim.t)
        # smpc.update(Sim.get_update_dict())
        # sol=smpc.solve()
        # Sim.step(sol["u_control"])
        Sim.step()