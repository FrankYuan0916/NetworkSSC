from b_networkSSC import *
from b_standardSSC import *
from b_adaptivessc import *

class TUNE:
    def __init__(self, data, label, dataname):
        self.data = data
        self.label = label
        self.pre_data = PREPROCESS(self.data, self.label)
        self.gene_num = self.pre_data.X.shape[0]
        self.dataname = dataname
        os.makedirs("model_result/{data}".format(data = self.dataname), exist_ok=True)
    
    def networkSSCTune(self, network_lambda1, network_lambda2, d, shuffle = False):        
        candidate = pd.DataFrame(self.expandgrid(network_lambda1, network_lambda2))
        candidate.columns = ['lambda_1', 'lambda_2']
        score = []
        index = []
        
        for j in tqdm(range(candidate.shape[0])):
            lambda_1 = candidate.iloc[j, 0]
            lambda_2 = candidate.iloc[j, 1]       
            data_ind = range(self.pre_data.data.shape[0])
            
            if self.gene_num <= 5000:
                self.pre_data.buildNetwork()
                network_ssc = NETWORKSSC(self.pre_data.X, self.label, self.pre_data.L)
                self.W, self.Q = network_ssc.networkSSC(lambda_1, lambda_2, self.pre_data.X, d)
            else:
                divide_num = self.pre_data.data.shape[0] // 5000 + 1
                group_num = self.pre_data.data.shape[0] // divide_num
                self.W = np.zeros((self.data.shape[1], self.data.shape[1]))
                
                for i in range(divide_num):
                    if shuffle:
                        if i == divide_num - 1:
                            data_part = self.pre_data.data.iloc[data_ind, :]
                        else:
                            rand_ind = np.random.choice(data_ind, group_num, replace=False)
                            data_part = self.pre_data.data.iloc[rand_ind, :]
                            data_ind = list(set(data_ind) - set(rand_ind))  
                    else:
                        if i == divide_num - 1:
                            data_part = self.pre_data.data.iloc[(group_num * i):self.pre_data.data.shape[0], :]
                        else:
                            data_part = self.pre_data.data.iloc[(group_num * i):(group_num * (i+1)), :]

                    self.pre_data_part = PREPROCESS(data_part, self.label)
                    self.pre_data_part.buildNetwork()
                    network_ssc = NETWORKSSC(data_part, self.label, self.pre_data_part.L)
                    self.W_part, self.Q_part = network_ssc.networkSSC(lambda_1, lambda_2, data_part, d)
                    self.W += self.W_part
                    if i == 0:
                        self.Q = self.Q_part.dot(data_part)
                    else:
                        self.Q += self.Q_part.dot(data_part)
                self.W = self.W / divide_num
                self.Q = self.Q / divide_num

            np.save("model_result/{data}/networkSSC_W_{lambda1}_{lambda2}_{d}.npy".format(data = self.dataname, lambda1 = str(lambda_1), lambda2 = str(lambda_2), d = str(d)), self.W)
            np.save("model_result/{data}/networkSSC_Q_{lambda1}_{lambda2}_{d}.npy".format(data = self.dataname, lambda1 = str(lambda_1), lambda2 = str(lambda_2), d = str(d)), self.Q)
            
            para_result = spectral_clustering(self.W, n_clusters = len(np.unique(self.label)), random_state = 42)
            np.save("model_result/{data}/networkSSC_cluster_{lambda1}_{lambda2}_{d}.npy".format(data = self.dataname, lambda1 = str(lambda_1), lambda2 = str(lambda_2), d = str(d)), para_result)
            score.append(normalized_mutual_info_score(self.label, para_result))
            index.append(adjusted_rand_score(self.label, para_result))
        self.network_result = pd.concat([candidate, pd.DataFrame(score), pd.DataFrame(index)], axis = 1)
        self.network_result.columns = ['lambda_1', 'lambda_2', 'NMI', 'RI']

        return self.network_result
    
    
    def standardSSCTune(self, ssc_lambda1, ssc_lambda2):
        candidate = pd.DataFrame(self.expandgrid(ssc_lambda1, ssc_lambda2))
        candidate.columns = ['alpha1', 'alpha2']

        score = []
        index = []

        for i in tqdm(range(candidate.shape[0])):
            alpha1 = candidate.iloc[i, 0]
            alpha2 = candidate.iloc[i, 1]
            ss = SSC(self.pre_data.X, self.label, alpha1=alpha1, alpha2=alpha2)
            Z = ss.computeCmat()
            W = np.abs(Z + np.transpose(Z))
            para_result = spectral_clustering(W, n_clusters = len(np.unique(self.label)))
            score.append(normalized_mutual_info_score(self.label, para_result))
            index.append(adjusted_rand_score(self.label, para_result))

            np.save("model_result/{data}/SSC_W_{lambda1}_{lambda2}.npy".format(data = self.dataname, lambda1 = str(alpha1), lambda2 = str(alpha2)), W)
            np.save("model_result/{data}/SSC_cluster_{lambda1}_{lambda2}.npy".format(data = self.dataname, lambda1 = str(alpha1), lambda2 = str(alpha2)), para_result)

        self.ssc_result = pd.concat([candidate, pd.DataFrame(score), pd.DataFrame(index)], axis = 1)
        self.ssc_result.columns = ['alpha1', 'alpha2', 'NMI', 'RI']
        return self.ssc_result
    
    def adaptiveSSCTune(self, adaptive_lambda):
        score = []
        index = []

        corr_matrix = np.corrcoef(self.pre_data.X, rowvar=False)
        corr_matrix = np.maximum(corr_matrix, np.zeros(corr_matrix.shape) + 1e-5)
        ada = ADAPTIVESSC(self.pre_data.X, self.label)

        for i in tqdm(range(len(adaptive_lambda))):
            W = ada.adaptiveSSC(self.pre_data.X, adaptive_lambda[i], corr_matrix)
            para_result = spectral_clustering(abs(W), n_clusters = len(np.unique(self.label)), random_state = 42)
            score.append(normalized_mutual_info_score(self.label, para_result))
            index.append(adjusted_rand_score(self.label, para_result))

            np.save("model_result/{data}/adaptiveSSC_W_{lambda1}.npy".format(data = self.dataname, lambda1 = str(adaptive_lambda[i])), abs(W))
            np.save("model_result/{data}/adaptiveSSC_cluster_{lambda1}.npy".format(data = self.dataname, lambda1 = str(adaptive_lambda[i])), para_result)

        self.adaptive_result = pd.concat([pd.DataFrame(adaptive_lambda), pd.DataFrame(score), pd.DataFrame(index)], axis = 1)
        self.adaptive_result.columns = ['alpha', 'NMI', 'RI']
        return self.adaptive_result

    
    def kmeansResult(self):
        kmeans_result = KMeans(n_clusters = len(np.unique(self.label)), random_state = 42).fit(np.transpose(self.pre_data.X)).labels_
        kmeans_nmi = normalized_mutual_info_score(self.label, kmeans_result)
        kmeans_ari = adjusted_rand_score(self.label, kmeans_result)
        return kmeans_nmi, kmeans_ari
    
    def extractMax(self, score_result, name): # name = "NMI" or "RI"
        return score_result.iloc[np.where(score_result[name] == max(score_result[name]))[0], :]
    
    def expandgrid(self, *itrs):
        product = list(itertools.product(*itrs))
        return {'Var{}'.format(i+1):[x[i] for x in product] for i in range(len(itrs))}