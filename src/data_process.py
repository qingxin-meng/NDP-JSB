import numpy as np
import pickle
import random
from pdb import set_trace
from numpy.linalg import inv
from scipy.sparse import csr_matrix



class data():
    def __init__(self,lambda_s_a=5,lambda_s_b=1, industry_dim=26, sample_threshold=5,
                 reg_position_sim=5e-5,reg_company_sim=5e-5,reg_location_sim=5e-5):
        self.lambda_s_a=lambda_s_a
        self.lambda_s_b=lambda_s_b
        self.indutry_dim=industry_dim
        self.read_data_from_file()
        self.sample_threshold=sample_threshold
        self.reg_position_sim=reg_position_sim
        self.reg_company_sim=reg_company_sim
        self.reg_location_sim=reg_location_sim
        return None

    def read_data_from_file(self):
        ## read raw data from file
        with open('./source_file/raw_data_log.p', 'rb') as f:
            self.raw_data = pickle.load(f)  ## it is a list, every row is [i,j,m,sl,su]
        with open('./source_file/the dict of W_in.p', 'rb') as f:
            self.W_in = pickle.load(f)  ## it is a dict
        with open('./source_file/the dict of voc_length.p', 'rb') as f:
            self.voc_length = pickle.load(f)  ## it is a dict
        with open('./source_file/topic_distribution_dict_5.p', 'rb') as f:
            self.varphi=pickle.load(f)
        with open('./source_file/topic_words_5.p', 'rb') as f:
            self.phi=pickle.load(f)
        with open('./source_file/side_feature.p','rb') as f:
            self.side_feature=pickle.load(f)
        with open('./source_file/words_count.p', 'rb') as f:
            self.words_count_matrix = pickle.load(f)

    def generate_fold(self, folds=5, lower_or_upper=0, shuffle=True, seed=100, construct_baseline_file=False):
        ## if lower_or_upper=0, lower bound for train, else training the upper bound
        self.lower_or_upper = lower_or_upper
        ## raw_data must be a list
        nr=len(self.raw_data)
        np.random.seed(seed)
        if shuffle is True:
            np.random.shuffle(self.raw_data)

        ## iterate the folds
        end_i=0
        fold=0
        while fold <folds:
            if nr//folds==0:
                start_i=end_i
                end_i=start_i+nr//folds
            else:
                start_i=end_i
                end_i=min(start_i + nr//folds +1,nr)
            self.raw_test=self.raw_data[start_i:end_i]
            self.raw_train=self.raw_data[:start_i]+self.raw_data[end_i:]

            iset=set()
            jset=set()
            for i,j,m,_,_, in self.raw_train:
                iset.add(i)
                jset.add(j)
            self.num_position = len(iset)
            self.num_company = len(jset)
            self.raw2inner_i={i:inner_i for inner_i,i in enumerate(iset)}
            self.raw2inner_j={j:inner_j for inner_j,j in enumerate(jset)}
            self.initiation()
            self.construct_inner_index(self.raw_train)
            self.compute_offset_and_M()
            # print('num_position:{},num_company:{},M:{},num_instance:{},min:{},max:{}'
            #       .format(self.num_position,self.num_company,self.M,nr,self.min,self.max))
            self.construct_train_set(self.raw_train)
            self.construct_inner_params()
            self.construct_salary_matrix(self.train_set)
            self.construct_test_set(self.raw_test)
            self.construct_side_feature_matrix(self.raw_train)
            self.construct_similarity_matrix()
            self.construct_reduced_T_for_tadw()
            # print('num_train:{},num_test:{}'.format(len(self.train_set),len(self.test_set)))

            if construct_baseline_file:
                self.construct_baseline_file(fold)
            fold += 1

            with open('./source_file/raw2inner_i.p','wb') as f:
                pickle.dump(self.raw2inner_i,f)
            with open('./source_file/raw2inner_j.p','wb') as f:
                pickle.dump(self.raw2inner_j,f)
            with open('./source_file/raw2inner_m.p','wb') as f:
                pickle.dump(self.raw2inner_m,f)

            yield self.num_position,self.num_company,self.S_matrix,self.X,self.X_ij,\
                  self.inner_varphi,self.phi,self.inner_W_in,self.inner_voc_length,\
                  self.inner_N_j,self.offset,self.M,self.train_set,self.test_set,\
                  self.i_jm,self.jm_i,self.j_i,self.ij_jm,self.j_jm,self.Y,self.min,self.max,\
                  self.inner_auxiliary_position, self.inner_auxiliary_company,self.b_position,self.b_company,\
                  self.T


    def generate_proportional_data(self, train_proportion=0.9, lower_or_upper=0,
                                   shuffle=True, seed=100,construct_baseline_file=False):
        ## if lower_or_upper=0, lower bound for train, else training the upper bound
        self.lower_or_upper = lower_or_upper
        ## raw_data must be a list
        nr=len(self.raw_data)
        np.random.seed(seed)
        if shuffle is True:
            np.random.shuffle(self.raw_data)

        train_num=int(np.ceil(nr*train_proportion))
        self.raw_train=self.raw_data[:train_num]
        self.raw_test=self.raw_data[train_num:]


        iset=set()
        jset=set()
        for i,j,m,_,_, in self.raw_train:
            iset.add(i)
            jset.add(j)
        self.num_position = len(iset)
        self.num_company = len(jset)
        self.raw2inner_i={i:inner_i for inner_i,i in enumerate(iset)}
        self.raw2inner_j={j:inner_j for inner_j,j in enumerate(jset)}
        self.initiation()
        self.construct_inner_index(self.raw_train)
        self.compute_offset_and_M()
        # print('num_position:{},num_company:{},M:{},num_instance:{},min:{},max:{}'
              # .format(self.num_position,self.num_company,self.M,nr,self.min,self.max))
        self.construct_train_set(self.raw_train)
        self.construct_inner_params()
        self.construct_salary_matrix(self.train_set)
        self.construct_test_set(self.raw_test)
        self.construct_side_feature_matrix(self.raw_train)
        self.construct_similarity_matrix()
        self.construct_reduced_T_for_tadw()
        # print('num_train:{},num_test:{}'.format(len(self.train_set), len(self.test_set)))

        with open('./source_file/raw2inner_i.p', 'wb') as f:
            pickle.dump(self.raw2inner_i, f)
        with open('./source_file/raw2inner_j.p', 'wb') as f:
            pickle.dump(self.raw2inner_j, f)
        with open('./source_file/raw2inner_m.p', 'wb') as f:
            pickle.dump(self.raw2inner_m, f)
        with open('./source_file/offset.p', 'wb') as f:
            pickle.dump(self.offset, f)
        with open('./source_file/raw_train.p', 'wb') as f:
            pickle.dump(self.raw_train, f)

        if construct_baseline_file:
            if self.lower_or_upper == 0:
                name = 'lower'
            else:
                name = 'upper'
            with open('train_data_' + name + '_trainpropotion{}.txt'.format(train_proportion), 'w') as f:
                for i, jm, s in self.train_set:
                    f.write(str(i) + '\t' + str(jm) + '\t' + str(s) + '\n')

            with open('test_data_' + name + '_trainpropotion{}.txt'.format(train_proportion), 'w') as f:
                for i, jm, s in self.test_set:
                    f.write(str(i) + '\t' + str(jm) + '\t' + str(s) + '\n')


        yield self.num_position,self.num_company,self.S_matrix,self.X,self.X_ij,\
              self.inner_varphi,self.phi,self.inner_W_in,self.inner_voc_length,\
              self.inner_N_j,self.offset,self.M,self.train_set,self.test_set,\
              self.i_jm,self.jm_i,self.j_i,self.ij_jm,self.j_jm,self.Y,self.min,self.max, \
              self.inner_auxiliary_position,self.inner_auxiliary_company,self.b_position,self.b_company, \
              self.T

    def generate_new_position_data(self, train_proportion=0.9, lower_or_upper=0,seed=100):
        ## if lower_or_upper=0, lower bound for train, else training the upper bound
        self.lower_or_upper = lower_or_upper
        ## raw_data must be a list
        nr=len(self.raw_data)
        test_num=np.floor(nr*(1-train_proportion))
        np.random.seed(seed)

        i_njm={}
        sample_set=set()
        for i, j, m, _, _, in self.raw_data:
            sample_set.add(i)
            if i not in i_njm:
                i_njm[i]=1
            else:
                i_njm[i]+=1

        total_sample=0
        select_set=set()
        while total_sample<test_num:
            select=random.sample(sample_set, 1)[0]
            if i_njm[select]>100:
                continue
            select_set.add(select)
            total_sample +=i_njm[select]
            sample_set.remove(select)

        raw_train=[]
        raw_test=[]
        for i, j, m, sl, su, in self.raw_data:
            if i in select_set:
                raw_test.append((i,j,m,sl,su))
            else:
                raw_train.append((i,j,m,sl,su))


        iset=set()
        jset=set()
        for i,j,m,_,_, in raw_train:
            iset.add(i)
            jset.add(j)
        self.num_position = len(iset)
        self.num_company = len(jset)
        self.raw2inner_i={i:inner_i for inner_i,i in enumerate(iset)}
        self.raw2inner_j={j:inner_j for inner_j,j in enumerate(jset)}
        self.initiation()
        self.construct_inner_index(raw_train)
        self.compute_offset_and_M()
        # print('num_position:{},num_company:{},M:{},num_instance:{},min:{},max:{}'
              # .format(self.num_position,self.num_company,self.M,nr,self.min,self.max))
        self.construct_train_set(raw_train)
        self.construct_inner_params()
        self.construct_salary_matrix(self.train_set)
        self.construct_test_set(raw_test,include_new_position=True)
        self.construct_side_feature_matrix(raw_train)
        self.construct_similarity_matrix()
        # print('num_train:{},num_test:{}'.format(len(self.train_set), len(self.test_set)))

        with open('./source_file/raw2inner_i.p', 'wb') as f:
            pickle.dump(self.raw2inner_i, f)
        with open('./source_file/raw2inner_j.p', 'wb') as f:
            pickle.dump(self.raw2inner_j, f)
        with open('./source_file/raw2inner_m.p', 'wb') as f:
            pickle.dump(self.raw2inner_m, f)
        with open('./source_file/offset.p', 'wb') as f:
            pickle.dump(self.offset, f)

        yield self.num_position,self.num_company,self.S_matrix,self.X,self.X_ij,\
              self.inner_varphi,self.phi,self.inner_W_in,self.inner_voc_length,\
              self.inner_N_j,self.offset,self.M,self.train_set,self.test_set,\
              self.i_jm,self.jm_i,self.j_i,self.ij_jm,self.j_jm,self.Y,self.min,self.max, \
              self.inner_auxiliary_position,self.inner_auxiliary_company,self.b_position,self.b_company

    def initiation(self):
        self.raw2inner_m = {}
        self.inner_N_j = {}
        self.inner_voc_length = {}
        self.inner_W_in = {}
        self.i_jm={}
        self.jm_i={}
        self.j_i={}
        self.ij_jm={}
        self.j_jm={}
        self.train_set = []
        self.test_set = []
        self.S_matrix=None
        self.offset=None
        self.M=None
        self.global_mean=None
        self.min=None
        self.max=None
        self.latent_dim = self.phi.shape[0]
        self.inner_varphi = np.zeros([self.latent_dim,self.num_position])


    def construct_inner_index(self,raw_train):
        current_m= np.zeros(self.num_company).astype(np.int)
        self.global_mean=0.0
        self.min=100
        self.max=-100
        n=0
        if self.lower_or_upper==0:
            for i,j,m,sl,su in raw_train:
                self.global_mean +=sl
                self.min=min(self.min,sl)
                self.max=max(self.max,sl)
                n +=1
                inner_j=self.raw2inner_j[j]
                if current_m[inner_j]==0:
                    self.raw2inner_m[inner_j]={}
                    self.raw2inner_m[inner_j][m]=0
                    self.inner_N_j[inner_j]=1
                    current_m[inner_j] +=1
                elif m not in  self.raw2inner_m[inner_j]:
                    self.raw2inner_m[inner_j][m] = current_m[inner_j]
                    current_m[inner_j] += 1
                    self.inner_N_j[inner_j] += 1
                elif m in self.raw2inner_m[inner_j]:
                    continue
        else:
            for i,j,m,sl,su in raw_train:
                self.global_mean +=su
                self.min = min(self.min, su)
                self.max = max(self.max, su)
                n +=1
                inner_j=self.raw2inner_j[j]
                if current_m[inner_j]==0:
                    self.raw2inner_m[inner_j]={}
                    self.raw2inner_m[inner_j][m]=0
                    self.inner_N_j[inner_j]=1
                    current_m[inner_j] +=1
                elif m not in  self.raw2inner_m[inner_j]:
                    self.raw2inner_m[inner_j][m] = current_m[inner_j]
                    current_m[inner_j] += 1
                    self.inner_N_j[inner_j] += 1
                elif m in self.raw2inner_m[inner_j]:
                    continue
        self.global_mean=self.global_mean/n
        self.global_mean=0.0
        self.min=self.min-self.global_mean
        self.max=self.max-self.global_mean



    def construct_train_set(self,raw_train):
        jm2jm = {key: jm for jm, key in enumerate(self.side_feature.keys())}
        self.raw_jm2inner_jm={}
        for i,j,m,sl,su in raw_train:
            inner_i = self.raw2inner_i[i]
            inner_j = self.raw2inner_j[j]
            inner_m = self.raw2inner_m[inner_j][m]
            jm=self.offset[inner_j]+inner_m


            if self.lower_or_upper == 0:
                self.train_set.append((inner_i, jm, sl))
            else:
                self.train_set.append((inner_i, jm, su))

            raw_jm = jm2jm[(j, m)]
            if raw_jm not in self.raw_jm2inner_jm:
                self.raw_jm2inner_jm[raw_jm] = jm

            if inner_i not in self.i_jm:
                self.i_jm[inner_i]=[jm]
            else:
                self.i_jm[inner_i].append(jm)
            if jm not in self.jm_i:
                self.jm_i[jm]=[inner_i]
            else:
                self.jm_i[jm].append(inner_i)
            if inner_j not in self.j_i:
                self.j_i[inner_j]=[inner_i]
            else:
                self.j_i[inner_j].append(inner_i)
            if (inner_i,inner_j) not in self.ij_jm:
                self.ij_jm[(inner_i,inner_j)]=[jm]
            else:
                self.ij_jm[(inner_i, inner_j)].append(jm)
            if inner_j not in self.j_jm:
                self.j_jm[inner_j]=[jm]
            elif jm not in self.j_jm[inner_j]:
                self.j_jm[inner_j].append(jm)
            else:
                continue

    def construct_test_set(self,raw_test, include_new_position=False):

        for i,j,m,sl,su in raw_test:
            if i not in self.raw2inner_i:
                if include_new_position is False:
                    continue
                else:
                    inner_i=self.varphi[i]
            else:
                inner_i = self.raw2inner_i[i]
            if j not in self.raw2inner_j:
                continue
            else:
                inner_j = self.raw2inner_j[j]
            if m not in self.raw2inner_m[inner_j]:
                continue
            else:
                inner_m=self.raw2inner_m[inner_j][m]
                jm=self.offset[inner_j]+inner_m

            if self.lower_or_upper==0:
                self.test_set.append((inner_i,jm,sl))
            else:
                self.test_set.append((inner_i,jm,su))

    def construct_inner_params(self):
        for i, ws in self.W_in.items():
            if i not in self.raw2inner_i:
                continue
            else:
                inner_i=self.raw2inner_i[i]
                self.inner_W_in[inner_i]=ws
        for i, length in self.voc_length.items():
            if i not in self.raw2inner_i:
                continue
            else:
                inner_i=self.raw2inner_i[i]
                self.inner_voc_length[inner_i]=length
        for i, var in self.varphi.items():
            if i not in self.raw2inner_i:
                continue
            else:
                inner_i=self.raw2inner_i[i]
                self.inner_varphi[:,inner_i]=var


    def compute_offset_and_M(self):
        self.M = 0
        self.offset = []
        for j in range(self.num_company):
            self.offset.append(self.M)
            self.M +=self.inner_N_j[j]
        return self.M,self.offset

    def construct_salary_matrix(self,train_set):
        self.S_matrix = np.zeros([self.num_position, self.M])
        self.X=np.zeros([self.num_position, self.M])
        self.X_ij=np.zeros([self.num_position, self.num_company])



        for inner_i, inner_jm, s in train_set:
            self.S_matrix[inner_i, inner_jm] = s
            self.X[inner_i,inner_jm]=self.lambda_s_a

        jm_ni = np.count_nonzero(self.S_matrix, axis=0)
        i_njm = np.count_nonzero(self.S_matrix, axis=1)
        full_set_position = set(range(self.num_position))
        full_set_M = set(range(self.M))

        S_mean_i = np.sum(self.S_matrix, 1) / i_njm
        self.b_position = S_mean_i
        self.S_matrix = self.S_matrix - self.b_position[:, None]
        self.S_matrix[np.where(self.X == 0)] = 0
        #
        S_mean_jm = np.sum(self.S_matrix, 0) / jm_ni
        self.b_company = S_mean_jm
        self.S_matrix = self.S_matrix - self.b_company[None, :]
        self.S_matrix[np.where(self.X == 0)] = 0




        for i in range(self.num_position):
            if i_njm[i]<self.sample_threshold:
                sample_size = 10
                res_set = full_set_M - set(self.i_jm[i])
                sample_j = random.sample(res_set, sample_size)
                for jj in sample_j:
                    self.X[i, jj] = self.lambda_s_b/sample_size
                    self.S_matrix[i,jj]=S_mean_i[i]-self.b_position[i]


        for jm in range(self.M):
            if jm_ni[jm] < self.sample_threshold:
                sample_size=10
                res_set = full_set_position - set(self.jm_i[jm])
                sample_i = random.sample(res_set, sample_size)
                for ii in sample_i:
                    self.X[ii,jm]=self.lambda_s_b/sample_size
                    self.S_matrix[ii,jm]=S_mean_jm[jm]-self.b_company[jm]


        for j in range(self.num_company):
            for m in range(self.inner_N_j[j]):
                jm = self.offset[j] + m
                self.X_ij[:, j] += self.X[:, jm]

        if sum(np.sum(self.X_ij,axis=0)==0)>0 or sum(np.sum(self.X_ij,axis=1)==0)>0:
            print('something wrong with X_ij, please cheak...')
            set_trace()

    def construct_baseline_file(self,fold):
        if self.lower_or_upper==0:
            name='lower'
        else:
            name='upper'
        with open('train_data_'+name+'_fold{}.txt'.format(fold),'w') as f:
                for i,jm,s in self.train_set:
                    f.write(str(i)+'\t'+str(jm)+'\t'+str(s)+'\n')

        with open('test_data_'+name+'_fold{}.txt'.format(fold),'w') as f:
                for i,jm,s in self.test_set:
                    f.write(str(i)+'\t'+str(jm)+'\t'+str(s)+'\n')

    def construct_side_feature_matrix(self,raw_train):
        self.Y = [np.zeros([self.M, self.indutry_dim]),
                  np.zeros([self.M, self.indutry_dim]),
                  np.zeros([self.M, 5]),
                  np.zeros([self.M, 6]),
                  np.zeros([self.M, 5]),
                  ]

        for i,j,m,sl,su in raw_train:
            inner_j = self.raw2inner_j[j]
            inner_m = self.raw2inner_m[inner_j][m]
            inner_jm=self.offset[inner_j]+inner_m
            for i in range(5):
                feature=self.side_feature[(j,m)][i]
                self.Y[i][inner_jm, feature]=1

    def construct_similarity_matrix(self):
        with open('./source_file/auxiliary_position.p', 'rb') as f:
            auxiliary_position=pickle.load(f)
        with open('./source_file/auxiliary_company.p', 'rb') as f:
            auxiliary_company=pickle.load(f)

        inner2raw_i = {inner_i: raw for raw, inner_i in self.raw2inner_i.items()}
        inner_jm2raw_jm={inner_jm: raw for raw, inner_jm in self.raw_jm2inner_jm.items()}


        self.inner_auxiliary_position = np.zeros((self.num_position, self.num_position))
        row_inner = np.repeat(range(self.num_position), self.num_position)
        col_inner = np.tile(range(self.num_position), self.num_position)
        raw_position_list = list(map(lambda x: inner2raw_i[x], range(self.num_position)))
        row = np.repeat(raw_position_list, self.num_position)
        col = np.tile(raw_position_list, self.num_position)
        self.inner_auxiliary_position[row_inner, col_inner] = auxiliary_position[row, col]

        self.inner_auxiliary_company = np.zeros((self.M, self.M))
        row_inner = np.repeat(range(self.M), self.M)
        col_inner = np.tile(range(self.M), self.M)
        raw_company_list = list(map(lambda x: inner_jm2raw_jm[x], range(self.M)))
        row = np.repeat(raw_company_list, self.M)
        col = np.tile(raw_company_list, self.M)
        self.inner_auxiliary_company[row_inner, col_inner] = auxiliary_company[row, col]

    def construct_reduced_T_for_tadw(self):
        with open('./source_file/reduced_T_for_tadw.p', 'rb') as f:
            T=pickle.load(f)

        inner2raw_i = {inner_i: raw for raw, inner_i in self.raw2inner_i.items()}
        raw_position_list = list(map(lambda x: inner2raw_i[x], range(self.num_position)))
        self.T=T[raw_position_list,:]