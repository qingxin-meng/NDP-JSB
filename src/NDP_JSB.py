## define the hyper-parameter alpha=1 which will be ignored during the estimation
import numpy as np
import pickle
import time
import random
from numpy.linalg import inv
from scipy.special import digamma
from scipy.special import beta as Beta
from scipy.special import loggamma as logGamma
from pdb import set_trace
from scipy.stats import pearsonr


def logmultivariate_beta(x):
    if isinstance(x,int):
        x=np.array([x])
    result = logGamma(x).sum()- logGamma(x.sum())
    return result

import scipy
scipy.special.loggamma


class optimization():
    def __init__(self, beta=1, omega=1, max_voc=5000, lambda_t=0.1,
                 lambda_s_a=1 , lambda_s_b=0.1, lambda_psi=1, latent_dim=5,
                 K=60, industry_dim=26, radius=1, update_varphi=False): 

        self.beta = beta
        self.omega = omega
        self.max_voc = max_voc
        self.lambda_s_a = lambda_s_a
        self.lambda_s_b = lambda_s_b
        self.lambda_t = lambda_t
        self.lambda_psi = lambda_psi
        self.latent_dim = latent_dim
        self.K = K
        self.radius = radius
        self.update_varphi = update_varphi
        self.industry_dim = industry_dim
        self.learning_rate =1.0
        self.old_mu_ = None
        self.old_T = None

    def initialization(self):
        ## initialize the latent variables.
        self.G = []
        for i in range(self.num_position):
            length = self.voc_length[i]
            temp = np.zeros([length, self.latent_dim])
            for n in range(length):
                w = self.W_in[i][n]
                temp[n, :] = self.varphi[:, i] * self.phi[:, w]
            temp = temp / np.sum(temp, axis=1)[:, None]
            self.G.append(temp)  # [I,length_i,L]

        self.theta_1 = np.ones([self.K])  # [K]
        self.theta_2 = np.ones([self.K]) * self.beta  # [K]
        self.Z =  np.random.random([self.M, self.K])
        self.Z = self.Z / (np.sum(self.Z, axis=1)[:, None])  # [M,K]
        self.mu_ = 0.01 * np.random.random([self.latent_dim, self.K])
        self.lambda_ = np.ones([self.latent_dim, self.K]) * self.lambda_psi  # [L,K]
        self.T = 0.01 * np.random.random([self.latent_dim, self.num_position  ])  # [L,I]

        self.delta = []
        self.delta.append(np.ones([self.industry_dim, self.K]) / self.industry_dim)
        self.delta.append(np.ones([self.industry_dim, self.K]) / self.industry_dim)
        self.delta.append(np.ones([5, self.K]) / 5.0)
        self.delta.append(np.ones([6, self.K]) / 6.0)
        self.delta.append(np.ones([5, self.K]) / 5.0)

    def train_predict(self, input, train_epoch=None):
        # set_trace()
        self.num_position, self.num_company, self.S, self.X, self.X_ij, \
        self.varphi, self.phi, self.W_in, self.voc_length, \
        self.N_j, self.offset, self.M, self.train_set, self.test_set, \
        self.i_jm, self.jm_i, self.j_i, self.ij_jm, self.j_jm, self.Y, self.min, self.max, \
        _, _, self.b_position, self.b_company, _ = input


        self.num_train_instance = len(self.train_set)
        if self.phi.shape[0] != self.latent_dim:
            print('the latent dim is not math, please check!')
            exit()
        self.initialization()
        print('training begining!\n')

        if train_epoch is None:
            self.old_loss = 1e5
            for epoch in range(50):
                self.s_time = time.time()
                self.one_epoch_updating(epoch)
                self.train_loss = 0.0
                train_rmse, train_mae, train_pr, train_pv = self.predict(self.train_set)
                test_rmse, test_mae, test_pr, test_pv, for_result_analysis = self.predict(self.test_set,
                                                                                          output_predictions=True)
                if self.old_loss - test_rmse < 1e-5:
                        time1 = time.time()
                        mins = (time1 - self.s_time) / 60
                        print('the train converges at epoch {}, need {:.2f} minutes. '
                              'the test rmse:{:.4f} mae:{:.4f},pr:{:.4f} pv:{:.4f},'
                              ' the train rmse:{:.4f}, the training loss:{:.4f}\n'.
                              format(epoch, mins, test_rmse, test_mae, test_pr, test_pv, train_rmse, self.full_loss))
                        break
                else:
                    self.old_loss = test_rmse
                    self.old_for_result_analysis = for_result_analysis
                    time1 = time.time()
                    mins = (time1 - self.s_time) / 60
                    print('epoch {}, need {:.2f} minutes. '
                          'the test rmse:{:.4f} mae:{:.4f},pr:{:.4f} pv:{:.4f},'
                          ' the train rmse:{:.4f}, the training loss:{:.4f}\n'.
                          format(epoch, mins, test_rmse, test_mae, test_pr, test_pv, train_rmse, self.full_loss))
                  
            else:
                for epoch in range(train_epoch):
                    self.s_time = time.time()
                    self.one_epoch_updating(epoch)
                    self.train_loss = 0.0
                    train_rmse, train_mae, train_pr, train_pv = self.predict(self.train_set)
                    test_rmse, test_mae, test_pr, test_pv = self.predict(self.test_set)
                    time1 = time.time()
                    mins = (time1 - self.s_time) / 60
                    print('epoch {}, need {:.2f} minutes. '
                          'the test rmse:{:.4f} mae:{:.4f}, pr:{:.4f} pv:{:.4f},'
                          'the train rmse:{:.4f}, the training loss:{:.4f}\n'.
                          format(epoch, mins, test_rmse, test_mae, test_pr, test_pv, train_rmse, self.full_loss))
                   
        else:
            for epoch in range(train_epoch):
                self.s_time = time.time()
                self.one_epoch_updating(epoch)
                self.train_loss = 0.0
                train_rmse, train_mae = self.predict(self.train_set)
                test_rmse, test_mae = self.predict(self.test_set)
                time1 = time.time()
                mins = (time1 - self.s_time) / 60
                print('epoch {}, need {:.2f} minutes. '
                      'the test rmse:{:.4f} mae:{:.4f}, the train rmse:{:.4f}, the training loss:{:.4f}\n'.
                      format(epoch, mins, test_rmse, test_mae, train_rmse,self.full_loss))
               

    def one_epoch_updating(self, epoch):
        if epoch<5:
            self.update_delta()
            self.update_theta()
        self.update_mu_lambda_matrix()
        self.update_T_matrix()
        # if epoch<5:
        self.update_Z_matrix()
        # self.compute_loss()
        self.full_loss=0

        ## updating ctr
        if self.update_varphi:
            self.phi, self.varphi, self.G = self.update_varphi_process(self.latent_dim, self.num_position,
                                                                       self.lambda_t,
                                                                       self.G, self.phi, self.T, self.varphi,
                                                                       self.voc_length,
                                                                       self.W_in)

    def compute_loss(self):
        
        self.num_clusters=self.K
        s = time.time()

        term_1=0.0
        for i in range(self.num_position):
            term_1 += -0.5*self.lambda_t*(self.T[:,i]-self.varphi[:,i]).T @ (self.T[:,i]-self.varphi[:,i])
        print('term_1',term_1, time.time()-s)
        s = time.time()
        
        # term_2=0.0
        # for i in range(self.num_position):
        #     for n in range(self.voc_length[i]):
        #         for l in range(self.latent_dim):
        #             w = self.W_in[i][n]
        #             term_2 += self.G[i][n,l]*(np.log(self.varphi[l,i]*self.phi[l,w]+1e-9)-np.log(self.G[i][n,l]))

        A = self.X * self.S * self.S  # [I,J]
        B = self.T.T @ self.mu_  # [I,K]
        C = self.X * self.S  # [I,J]
        term_3 = -0.5 * (A @ self.Z).sum() + (B.T @ C * self.Z.T).sum() - 0.5 * (self.X @ self.Z * self.TrhoT).sum()
        print('term_3', term_3, time.time() - s)
        s = time.time()


        
        temp = digamma(self.theta_1 + self.theta_2)
        Exception_1 = digamma(self.theta_1) - temp  # [K]
        Exception_2 = digamma(self.theta_2) - temp  # [K]

        term_4=0.0
        for j in range(self.M):
            for k in range(self.num_clusters):
                z_jg=0.0
                for g in range(k+1,self.num_clusters):
                    z_jg += self.Z[j,g]
                term_4 += z_jg*Exception_2[k]+self.Z[j,k]*Exception_1[k]
        print('term_4', term_4, time.time() - s)
        s = time.time()

        term_5=0.0
        for k in range(self.num_clusters):
            term_5 += np.log(self.beta)+(self.beta-1)*Exception_2[k]
        print('term_5', term_5, time.time() - s)
        s = time.time()

        term_6=0.0
        for k in range(self.num_clusters):
            term_6 += 0.5* self.latent_dim*np.log(self.lambda_psi/2/3.1415+1e-9)\
            -0.5*self.lambda_psi*(self.mu_[:,k].T@self.mu_[:,k]+(1.0/self.lambda_[:,k]).sum())
        print('term_6', term_6, time.time() - s)
        s = time.time()
        
        term_7=0.0

        ## be careful, needs to be re-check!
        Expectation_of_log_delta = np.zeros([self.M, self.K])
        for d in range(5):
            Expectation_of_log_delta += self.Y[d] @ (
                    digamma(self.delta[d]) - digamma(np.sum(self.delta[d], axis=0))[None, :])  # [M,K]

        for j in range(self.M):
            for k in range(self.num_clusters):
                    term_7 += self.Z[j,k]*Expectation_of_log_delta[j,k]
        print('term_7', term_7, time.time() - s)
        s = time.time()

        term_8=0.0
        self.dim_d=[self.industry_dim,self.industry_dim,5,6,5]

        self.Expectation_of_log_delta_3D = [np.zeros([self.industry_dim,self.K]),
                                            np.zeros([self.industry_dim,self.K]),
                                            np.zeros([5,self.K]),
                                            np.zeros([6,self.K]),
                                            np.zeros([5,self.K])]   #[D=5,dim_d, K]
        for d in range(5):
            self.Expectation_of_log_delta_3D[d] += digamma(self.delta[d]) \
            - digamma(np.sum(self.delta[d], axis=0))[None,:]  # [D=5, dim_d, K]

        for k in range(self.num_clusters):
            for d in range(5):
                term_8 += (self.omega-1)*self.Expectation_of_log_delta_3D[d][:,k].sum()
        print('term_8', term_8, time.time() - s)
        s = time.time()


        term_9=0.0
        for k in range(self.num_clusters):
            term_9 -= 0.5* np.log((self.lambda_[:,k]/2/3.1415).sum()+1e-9)-0.5*self.num_clusters

        print('term_9', term_9, time.time() - s)
        s = time.time()

        term_10=0.0
        for j in range(self.M):
            for k in range(self.num_clusters):
                term_10 -= self.Z[j,k]*np.log(self.Z[j,k]+1e-9)

        print('term_10', term_10, time.time() - s)
        s = time.time()


        term_11=0.0
        for k in range(self.num_clusters):
            term_11 -= -np.log(Beta(self.theta_1[k],self.theta_2[k])+1e-9) \
                        +(self.theta_1[k]-1)*Exception_1[k]+(self.theta_2[k]-1)*Exception_2[k]
        print('term_11', term_11, time.time() - s)
        s = time.time()

        term_12=0.0
        for d in range(5):
            for k in range(self.num_clusters):
                term_12 -= ((self.delta[d]-1)[:,k] * self.Expectation_of_log_delta_3D[d][:,k]).sum()\
                            - logmultivariate_beta(self.delta[d][:,k])
        print('term_12', term_12, time.time() - s)
        s = time.time()

       

        self.full_loss = -(term_1+term_3 + term_4 + term_5 + term_6 \
                           + term_7 + term_8 + term_9 + term_10 + term_11 + term_12)
    
        if self.old_mu_ is not None:
            print('delta mu', np.abs(self.old_mu_-self.mu_).mean())
            print('delta T', np.abs(self.old_T - self.T).mean())

        self.old_mu_ = self.mu_.copy()
        self.old_T = self.T.copy()

        return self.full_loss

    def update_mu_lambda_sgd(self):
        ## updating mu_ and lambda_
        t_feature = self.T.take(
            [int(i) for i, jm, s in self.train_set], axis=1
        )  # [L,N]
        z_jm = self.Z.take(
            [int(jm) for i, jm, s in self.train_set], axis=0
        )  # [N,K]
        s_feature = np.array([s for i, jm, s in self.train_set])  # [N]
        up = t_feature @ (s_feature[:, None] * z_jm)  # [L,K]
        for k in range(self.K):
            down = t_feature @ (t_feature * z_jm[:, k][None, :]).T * self.lambda_s_a #+ self.lambda_psi  # [L,L]
            down = down + self.lambda_psi*np.eye(down.shape)
            self.mu_[:, k] = inv(down) @ up[:, k]
            self.lambda_[:, k] = np.sum(t_feature ** 2 * z_jm[:, k][None, :] * self.lambda_s_a,
                                        axis=1) + self.lambda_psi
        time1 = time.time()
        mins = (time1 - self.s_time) / 60
        # print('updating mu and lambda need time {:.2f}\n'.format(mins))

    def update_mu_lambda_matrix(self):
        ## updating mu_ and lambda_
        XZ = self.X @ self.Z  # [I,K]
        TT_ = self.T ** 2
        term1 = self.T @ ((self.X * self.S) @ self.Z)  # [L,K]
        for k in range(self.K):
            try:
                down = self.T @ (XZ[:, k][:, None] * self.T.T)
                down += self.lambda_psi*np.eye(*down.shape)
                down = inv(down)
            except:
                set_trace()
            up = term1[:, k]  # [L]
            self.mu_[:, k] = (1-self.learning_rate)* self.mu_[:,k] + self.learning_rate *down @ up  # [L]
        self.lambda_ = (1-self.learning_rate)*self.lambda_ + \
                       self.learning_rate*(TT_ @ self.X @ self.Z + self.lambda_psi)  # [L,K]
        time1 = time.time()
        mins = (time1 - self.s_time) / 60

    def update_theta(self):
        ## updating theta
        self.theta_1 = 1 + np.sum(self.Z, axis=0)  # [K]
        cumsum = np.cumsum(self.Z, axis=1)  # [M,K]
        self.theta_2 = self.beta + np.sum((1 - cumsum), axis=0)  # [K]
        time1 = time.time()
        mins = (time1 - self.s_time) / 60

    def update_delta(self):
        ## updating delta
        for i in range(5):
            try:
                self.delta[i] = self.omega + self.Y[i].T @ self.Z  ##[V,K]
            except:
                set_trace()
        time1 = time.time()
        mins = (time1 - self.s_time) / 60


    def update_T_matrix(self):
        XZ = self.X @ self.Z  # (J,C) (C,K) -> (J,K)
        inv_lambda_ = 1.0 / self.lambda_  # [L,K]
        up_T = (self.mu_ @ self.Z.T) @ (self.X * self.S).T + self.lambda_t * self.varphi  # [L,I]

        temp_inv_lambda_ = XZ @ inv_lambda_.T  # J*L
        for i in range(self.num_position):
            term = self.mu_ @ np.diagflat(XZ[i]) @ self.mu_.T  # L*L
            term += np.diagflat(temp_inv_lambda_[i])
            down_T = inv(term + self.lambda_t * np.eye(*term.shape))
            self.T[:, i] = (1-self.learning_rate)*self.T[:,i] + self.learning_rate*down_T @ up_T[:, i]




    def update_Z_matrix(self):
        ## updating Z
        temp = digamma(self.theta_1 + self.theta_2)
        Exception_1 = digamma(self.theta_1) - temp  # [K]
        Exception_2 = digamma(self.theta_2) - temp  # [K]
        Exception_2_ = np.cumsum(Exception_2)  # [K]
        Exception_2_ = np.append(0, Exception_2_[:-1])  # [K]
        Tmu = self.T.T @ self.mu_  # [I,K]

        self.TrhoT = np.zeros([self.num_position, self.K])  # [I,K]

        #####################
        inv_lambda_ = 1.0 / self.lambda_  # [L,K]
        self.rho = np.zeros([self.latent_dim, self.latent_dim, self.K])
        for k in range(self.K):
            MuMuk = self.mu_[:, k][None, :].T @ np.transpose(self.mu_[:, k])[None, :]
            Tri_lambda = np.diagflat(inv_lambda_[:, k])
            self.rho[:, :, k] = MuMuk + Tri_lambda
        #####################

        for i in range(self.num_position):
            for k in range(self.K):
                self.TrhoT[i, k] = self.T[:, i].T @ self.rho[:, :, k] @ self.T[:, i]  # [1]

        term1 = np.sum(self.X * (self.S ** 2), axis=0)[:, None]  # [M,1]
        term2 = (self.X * self.S).T @ Tmu  # [M,K]
        term3 = self.X.T @ self.TrhoT  # [M,K]
        term4 = -0.5 * (term1 - 2 * term2 + term3)

        Expectation_of_log_delta = np.zeros([self.M, self.K])
        for i in range(5):
            Expectation_of_log_delta += self.Y[i] @ (
                    digamma(self.delta[i]) - digamma(np.sum(self.delta[i], axis=0))[None, :])  # [M,K]

        zz = Exception_1[None, :] + Exception_2_[None, :] + term4 + Expectation_of_log_delta
        offset_zz = np.max(zz, axis=1)[:, None]
        self.Z = np.exp(zz - offset_zz)
        self.Z = self.Z / np.sum(self.Z, axis=1)[:, None]  # [M,K]
        time1 = time.time()
        mins = (time1 - self.s_time) / 60
        # set_trace()
        # print('updating Z need time {:.2f}\n'.format(mins))

    def predict(self, test_data, output_predictions=False):
        rmse = 0.0
        mae = 0.0
        num = 0.0
   

        for_result_analyze = []
        for i, jm, s in test_data:
            s = s
            # s = s*0.1
            s_predict = 0.0
            if isinstance(i, int):
                # if self.b_position is not None:
                #     s_predict += self.b_position[i]
                # if self.b_company is not None:
                #     s_predict += self.b_company[jm]
                s_predict += np.sum(self.T[:, i] * (self.mu_ @ self.Z[jm, :]))
                # s_predict += np.sum(self.T[:, i] * (self.mu_ @ Z[jm, :]))
                b_position=self.b_position[i]

            # if isinstance(i, list) :
            else: #isinstance(i, np.array)
                current_varphi=i
                diss=(current_varphi/np.sum(current_varphi))@(self.varphi/np.sum(self.varphi,axis=0)[None,:]) # currentvarphi 与训练集中varphi的距离
                # set_trace()
                sort_ids = np.argsort(diss)[::-1][:50]
                T=self.varphi[:,sort_ids].mean(axis=1)
                b_position = np.mean(self.b_position[sort_ids])
                s_predict += np.sum(T* (self.mu_ @ self.Z[jm, :]))
           
            s_predict= s_predict+b_position+self.b_company[jm]
            rmse += (s_predict - s) ** 2
            mae += np.abs(s_predict - s)
            for_result_analyze.append((i, jm, s, s_predict))
            num += 1
        ss = [r[2] for r in for_result_analyze]
        s_predicts = [r[3] for r in for_result_analyze]
        pr, pv = pearsonr(ss, s_predicts)
        rmse = np.sqrt(rmse / num)
        mae = mae / num
        if output_predictions == False:
            return rmse, mae, pr, pv
        else:
            return rmse, mae, pr, pv, for_result_analyze


    def update_varphi_process(self, latent_dim, num_position, lambda_t, G, phi, T, varphi, voc_length, W_in, radius=1):

        time1 = time.time()
        loglikelihood_ctr = 0.0
        new_varphi = np.zeros_like(varphi)
        new_G = []

        ## updating varhpi
        for i in range(num_position):
            Gi = G[i]
            sumn_Gi = np.sum(Gi, axis=0)  # [L]
            varphi_i_old = varphi[:, i]
            Ti = T[:, i]
            varphi_i = self.update_varphi_i(lambda_t, varphi_i_old, sumn_Gi, Ti, radius)
            new_varphi[:, i] = varphi_i

        for i in range(num_position):
            Gi = G[i]
            for n in range(voc_length[i]):
                ## updating phi
                w = W_in[i][n]
                phi[:, w] += Gi[n, :]
                ## updating G
                Gi[n, :] = new_varphi[:, i] * phi[:, w]
            Gi = Gi / np.sum(Gi, axis=1)[:, None]
            new_G.append(Gi)
        phi = phi / np.sum(phi, axis=1)[:, None]

        ## compute the loglikelihood_ctr
        # for i in range(num_position):
        #     Gi = G[i]
        #     for n in range(voc_length[i]):
        #         w = W_in[i][n]
        #         for l in range(latent_dim):
        #             if (1e-9 + new_varphi[l, i] * phi[l, w]) < 0 or (1e-9 + Gi[n, l]) < 0:
        #                 set_trace()
        #             loglikelihood_ctr += Gi[n, l] * (
        #                         np.log(1e-9 + new_varphi[l, i] * phi[l, w]) - np.log(1e-9 + Gi[n, l]))
        # print('loglikelihood_ctr:{:.10e}'.format(loglikelihood_ctr))
        # print('updating ctr need time:{} min'.format((time.time() - time1) / 60))
        return phi, new_varphi, new_G

    def update_varphi_i(self, lambda_t, varphi_i, sumn_Gi, Ti, radius):

        converge = False
        for round in range(50):
            if not converge:
                varphi_i_old = varphi_i.copy()  # [L]
                f_old = self.f_simplex(lambda_t, varphi_i_old, sumn_Gi, Ti)  # [1]
                gradient = self.df_simplex(lambda_t, varphi_i_old, sumn_Gi, Ti)  # [L]
                gradient_sum = np.sum(np.abs(gradient))  # [1]
                if gradient_sum > 1.0:
                    gradient = gradient / gradient_sum
                varphi_i -= gradient
                varphi_i_on_simplex = self.simplex_projection(varphi_i, radius)  # [L]
                varphi_i_on_simplex = varphi_i_on_simplex - varphi_i_old
                r = 0.5 * np.transpose(varphi_i_on_simplex) @ gradient  # [1]

                beta = 0.5
                for epoch in range(10):
                    varphi_i = varphi_i_old.copy()
                    varphi_i += beta * varphi_i_on_simplex
                    f_new = self.f_simplex(lambda_t, varphi_i, sumn_Gi, Ti)
                    if f_new > (f_old + beta * r):
                        beta = beta * 0.5
                    else:
                        converge = True
                        break
            else:
                break
        if not self.is_feasible(varphi_i):
            print('something is wrong about varphi_i, please check!')
        return varphi_i

    def is_feasible(self, x):
        sum = 0.0
        for val in x:
            if val < 0 or val > 1:
                return False
            sum += val
        if sum - 1 > 1e-5:
            return False
        else:
            return True

    def f_simplex(self, lambda_t, varphi_i, sumn_Gi, Ti):
        f_1 = -lambda_t * np.sum((Ti - varphi_i) ** 2)
        f_2 = np.sum(sumn_Gi * np.log(varphi_i))
        f = -(f_1 + f_2)
        return f

    def df_simplex(self, lambda_t, varphi_i, sumn_Gi, Ti):
        gradient = -lambda_t * (Ti - varphi_i) + sumn_Gi / varphi_i  # [L]
        return -gradient  # [L]

    def simplex_projection(self, x, radius):
        size = len(x)
        proj_x = np.zeros([size])
        cumsum = -radius
        sort_idx = np.argsort(x)[::-1]
        j = 0
        for idx in sort_idx:
            u = x[idx]
            cumsum += u
            if u > cumsum / (j + 1):
                j += 1
            else:
                break
        theta = cumsum / j
        for i in range(size):
            u = x[i] - theta
            if u <= 0:
                u = 0.0
            proj_x[i] = u
        proj_x = proj_x / np.sum(proj_x)
        return proj_x
