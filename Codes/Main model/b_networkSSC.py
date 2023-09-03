from a_preData import *

class NETWORKSSC:
    def __init__(self, data, label, L):
        self.X = data
        self.label = label
        self.L = L
        
    def f_1(self, Q, Z):                    # First Term
        return (1/2) * (np.linalg.norm(np.dot(Q, self.X) - np.dot(np.dot(Q, self.X), Z))) ** 2

    def f_2(self, Z, lambda_1):             # Second Term
        return lambda_1 * np.linalg.norm(Z, ord = 1)

    def f_3(self, Q, lambda_2):             # Third Term
        return (1/2) * lambda_2 * np.trace(np.dot(np.dot(Q, self.L), np.transpose(Q)))

    def f(self, Q, Z, lambda_1, lambda_2):  # Complete
        return self.f_1(Q, Z) + self.f_2(Z, lambda_1 = lambda_1) + self.f_3(Q, lambda_2 = lambda_2)

    def G(self, Q, Z):
        return np.dot(np.transpose(-np.dot(Q, self.X)), np.dot(Q, self.X) - np.dot(np.dot(Q, self.X), Z))

    def C(self, Q, Z, mu):
        return Z - self.G(Q, Z) / mu
    
    def networkSSC(self, lambda_1, lambda_2, X, d = 100):
        m = X.shape[1]
        lambda_3 = 1
        # Initialization
        # np.random.seed(42)
        Q = PCA(n_components = d).fit(np.transpose(X)).components_
        Z = np.dot(np.linalg.inv(np.dot(np.transpose(np.dot(Q, X)), np.dot(Q, X)) + lambda_3 * np.identity(m)), np.dot(np.transpose(np.dot(Q, X)), np.dot(Q, X)))
        Z = Z - np.diag(np.diag(Z))
        mu = 100 * (np.linalg.norm(np.dot(Q, X), ord = 'nuc')) ** 2
        
        # Iteration
        itr = 0
        while True:
            # Store the initial value
            Z0 = Z
            Q0 = Q

            # Update Z
            Z = np.maximum(self.C(Q, Z, mu = mu) - lambda_1/mu, 0) + np.minimum(self.C(Q, Z, mu = mu) + lambda_1/mu, 0)
            Z = Z - np.diag(np.diag(Z))

            # Update Q
            H = X - np.dot(X, Z)
            M = np.dot(H, np.transpose(H)) + lambda_2 * self.L
            # V = np.linalg.svd(M)[2]
            # Q = np.transpose(V[:, (M.shape[1] - d):])
            Q = np.transpose(scipy.linalg.eigh(M, eigvals = (0, d - 1))[1])

            # Update mu
            mu = 100 * (np.linalg.norm(np.dot(Q, X), ord = 'nuc')) ** 2


            itr += 1
            # print(np.linalg.norm(Z - Z0) / np.linalg.norm(Z0))
            # print(abs(self.f(Q, Z, lambda_1 = lambda_1, lambda_2 = lambda_2) - self.f(Q0, Z0, lambda_1 = lambda_1, lambda_2 = lambda_2)))
            # print(Z)
            if np.linalg.norm(Z - Z0) / np.linalg.norm(Z0) <= 1e-2 and abs(self.f(Q, Z, lambda_1 = lambda_1, lambda_2 = lambda_2) - self.f(Q0, Z0, lambda_1 = lambda_1, lambda_2 = lambda_2)) <= 1e-2:
                break
            if itr > 1000:
                print('-'*10 + 'lambda_1 = ' + str(lambda_1) + ', lambda_2 = ' + str(lambda_2) + ', Warning: Not Converge!' + '-'*10)
                break
        
        # Return Symmetric Matrix W
        W = (1/2) * (abs(Z) + abs(np.transpose(Z)))
        return W, Q
        
        
        